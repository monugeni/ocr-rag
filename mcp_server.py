#!/usr/bin/env python3
"""
Document RAG MCP Server
========================
Single server for all folders. Every search requires a folder path.

Tools:
    Discovery:
        list_folders        - List all folders
        list_folder_entries - List folders/files in a folder
        list_documents      - List documents in a folder scope
        get_document_info   - Full metadata for a document
        get_toc             - Section tree for a document

    Search:
        search_pages        - FTS keyword search with abbreviation expansion + OR fallback
        search_sections     - Search section headings
        semantic_search     - Vector/embedding search for conceptual similarity
        get_section         - Get all pages belonging to a section heading

    Navigation:
        get_page            - Get full page content with context
        get_pages           - Get a range of pages
        get_adjacent        - Get next/previous page

    Fallback (LLM calls these when Marker output looks wrong):
        reextract_page      - Re-extract page text from original PDF via pdfplumber
        reextract_table     - Re-extract table from original PDF via pdfplumber

Usage:
    python mcp_server.py --db docs.db --port 8200
"""

import argparse
import json
import re
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from corrections import register_correction_tools


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH = "docs.db"
ToolResult = dict[str, Any] | list[dict[str, Any]]


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def sanitize_fts(query: str) -> str:
    """Sanitize query for FTS5. Handles special characters, section numbers like 3.2."""
    words = query.strip().split()
    out = []
    for w in words:
        if w.upper() in ('AND', 'OR', 'NOT', 'NEAR'):
            continue
        # Section/clause numbers like "3.2", "B31.3" — quote as phrase so
        # the tokenizer splits internally but FTS treats them as adjacent tokens
        if re.search(r'\w\.\w', w):
            clean = re.sub(r'[^\w.]+', ' ', w).strip()
            if clean:
                out.append(f'"{clean}"')
        # Words with special chars — split on delimiters and quote parts
        elif re.search(r'[-/\\@#$%&*()+=,:;^~\[\]{}!?<>]', w):
            parts = re.split(r'[-/\\.,;:]+', w)
            out.extend(f'"{p.strip()}"' for p in parts if p.strip())
        else:
            w = w.strip('"\'')
            if w:
                out.append(w)
    return ' '.join(out)


# ---------------------------------------------------------------------------
# Abbreviation / synonym expansion
# ---------------------------------------------------------------------------

ABBREVIATIONS = {
    # ===== MATERIALS & METALLURGY =====
    'MOC': ['material', 'construction'],
    'CS': ['carbon', 'steel'],
    'SS': ['stainless', 'steel'],
    'AS': ['alloy', 'steel'],
    'LAS': ['low', 'alloy', 'steel'],
    'HAS': ['high', 'alloy', 'steel'],
    'HSLA': ['high', 'strength', 'low', 'alloy'],
    'GI': ['galvanized', 'iron'],
    'CI': ['cast', 'iron'],
    'DI': ['ductile', 'iron'],
    'MS': ['mild', 'steel'],
    'LTCS': ['low', 'temperature', 'carbon', 'steel'],
    'DSS': ['duplex', 'stainless', 'steel'],
    'SDSS': ['super', 'duplex', 'stainless', 'steel'],
    'CRA': ['corrosion', 'resistant', 'alloy'],
    'FRP': ['fibre', 'reinforced', 'plastic'],
    'GRP': ['glass', 'reinforced', 'plastic'],
    'HDPE': ['high', 'density', 'polyethylene'],
    'PP': ['polypropylene'],
    'PTFE': ['polytetrafluoroethylene', 'teflon'],
    'SS304': ['stainless', 'steel', '304'],
    'SS316': ['stainless', 'steel', '316'],
    'SS304L': ['stainless', 'steel', '304L'],
    'SS316L': ['stainless', 'steel', '316L'],
    'NACE': ['national', 'association', 'corrosion', 'engineers'],
    'BHN': ['brinell', 'hardness', 'number'],
    'HRC': ['hardness', 'rockwell'],
    'CVN': ['charpy', 'notch', 'impact'],
    'SMLS': ['seamless'],
    'ERW': ['electric', 'resistance', 'welded'],
    'SAW': ['submerged', 'arc', 'welded'],
    'EFW': ['electric', 'fusion', 'welded'],
    'SCH': ['schedule'],

    # ===== DIMENSIONS & MEASUREMENTS =====
    'OD': ['outside', 'diameter'],
    'ID': ['inside', 'diameter'],
    'NB': ['nominal', 'bore'],
    'NPS': ['nominal', 'pipe', 'size'],
    'DN': ['diameter', 'nominal'],
    'WT': ['wall', 'thickness'],
    'CL': ['class', 'rating'],
    'THK': ['thickness'],
    'LG': ['length'],
    'DIA': ['diameter'],
    'SWG': ['standard', 'wire', 'gauge'],
    'BWG': ['birmingham', 'wire', 'gauge'],
    'RL': ['random', 'length'],
    'TBE': ['threaded', 'both', 'ends'],
    'POE': ['plain', 'one', 'end'],
    'PBE': ['plain', 'both', 'ends'],
    'BW': ['butt', 'weld'],
    'SW': ['socket', 'weld'],
    'RF': ['raised', 'face'],
    'FF': ['flat', 'face'],
    'RTJ': ['ring', 'type', 'joint'],
    'BOM': ['bill', 'materials'],

    # ===== WELDING & FABRICATION =====
    'WPS': ['welding', 'procedure', 'specification'],
    'PQR': ['procedure', 'qualification', 'record'],
    'WPQ': ['welder', 'performance', 'qualification'],
    'WPQR': ['welding', 'procedure', 'qualification', 'record'],
    'SMAW': ['shielded', 'metal', 'arc', 'welding'],
    'GTAW': ['gas', 'tungsten', 'arc', 'welding'],
    'TIG': ['tungsten', 'inert', 'gas', 'welding'],
    'MIG': ['metal', 'inert', 'gas', 'welding'],
    'GMAW': ['gas', 'metal', 'arc', 'welding'],
    'FCAW': ['flux', 'cored', 'arc', 'welding'],
    'PWHT': ['post', 'weld', 'heat', 'treatment'],
    'HT': ['heat', 'treatment'],
    'SR': ['stress', 'relieving'],
    'PWSR': ['post', 'weld', 'stress', 'relief'],
    'HAZ': ['heat', 'affected', 'zone'],
    'WML': ['weld', 'metal'],
    'PMI': ['positive', 'material', 'identification'],
    'DWG': ['drawing'],
    'GA': ['general', 'arrangement'],
    'BBS': ['bar', 'bending', 'schedule'],
    'MTO': ['material', 'take', 'off'],

    # ===== INSPECTION & TESTING (NDE/NDT) =====
    'NDE': ['non', 'destructive', 'examination'],
    'NDT': ['non', 'destructive', 'testing'],
    'RT': ['radiographic', 'testing', 'radiography'],
    'UT': ['ultrasonic', 'testing'],
    'MT': ['magnetic', 'particle', 'testing'],
    'PT': ['penetrant', 'testing'],
    'DP': ['dye', 'penetrant'],
    'LPT': ['liquid', 'penetrant', 'testing'],
    'VT': ['visual', 'testing', 'inspection'],
    'TOFD': ['time', 'flight', 'diffraction'],
    'PAUT': ['phased', 'array', 'ultrasonic'],
    'AUT': ['automated', 'ultrasonic', 'testing'],
    'MFL': ['magnetic', 'flux', 'leakage'],
    'ET': ['eddy', 'current', 'testing'],
    'HT_TEST': ['hydrostatic', 'test', 'hydrotest'],
    'HYDRO': ['hydrostatic', 'test', 'hydrotest'],
    'AT': ['acceptance', 'test'],
    'MTC': ['material', 'test', 'certificate'],
    'CMTR': ['certified', 'material', 'test', 'report'],
    'TC': ['test', 'certificate'],
    'COC': ['certificate', 'conformity'],
    'MR': ['material', 'requisition'],
    'IRN': ['inspection', 'release', 'note'],
    'ITP': ['inspection', 'test', 'plan'],
    'QAP': ['quality', 'assurance', 'plan'],
    'QCP': ['quality', 'control', 'plan'],
    'QMS': ['quality', 'management', 'system'],
    'TPI': ['third', 'party', 'inspection'],
    'FAT': ['factory', 'acceptance', 'test'],
    'SAT': ['site', 'acceptance', 'test'],
    'SIT': ['site', 'inspection', 'test'],
    'FSAT': ['final', 'site', 'acceptance', 'test'],
    'PDI': ['pre', 'dispatch', 'inspection'],
    'PPI': ['pre', 'production', 'inspection'],
    'MDR': ['manufacturer', 'data', 'report'],
    'DFR': ['daily', 'field', 'report'],
    'NCR': ['non', 'conformance', 'report'],
    'CAPA': ['corrective', 'action', 'preventive', 'action'],
    'RFI': ['request', 'inspection'],
    'FDR': ['final', 'documentation', 'report'],

    # ===== PIPING & VALVES =====
    'PID': ['piping', 'instrumentation', 'diagram'],
    'PFD': ['process', 'flow', 'diagram'],
    'ISO': ['isometric', 'drawing'],
    'MOV': ['motor', 'operated', 'valve'],
    'AOV': ['air', 'operated', 'valve'],
    'SOV': ['solenoid', 'operated', 'valve'],
    'HOV': ['hand', 'operated', 'valve'],
    'PSV': ['pressure', 'safety', 'valve'],
    'PRV': ['pressure', 'relief', 'valve'],
    'BDV': ['blowdown', 'valve'],
    'SDV': ['shutdown', 'valve'],
    'CSV': ['car', 'sealed', 'valve'],
    'CV': ['control', 'valve'],
    'NRV': ['non', 'return', 'valve'],
    'BFV': ['butterfly', 'valve'],
    'BV': ['ball', 'valve'],
    'GV': ['gate', 'valve'],
    'GLV': ['globe', 'valve'],
    'DBB': ['double', 'block', 'bleed'],
    'ESDV': ['emergency', 'shutdown', 'valve'],
    'RO': ['restriction', 'orifice'],
    'FO': ['figure', 'eight', 'blind'],
    'SB': ['spectacle', 'blind'],

    # ===== EQUIPMENT — STATIC =====
    'HE': ['heat', 'exchanger'],
    'STHE': ['shell', 'tube', 'heat', 'exchanger'],
    'ACHE': ['air', 'cooled', 'heat', 'exchanger'],
    'PHE': ['plate', 'heat', 'exchanger'],
    'WHR': ['waste', 'heat', 'recovery'],
    'WHRB': ['waste', 'heat', 'recovery', 'boiler'],
    'HRSG': ['heat', 'recovery', 'steam', 'generator'],
    'PV': ['pressure', 'vessel'],
    'SG': ['steam', 'generator'],
    'LSG': ['lower', 'steam', 'generator'],
    'USG': ['upper', 'steam', 'generator'],
    'SH': ['superheater'],
    'RH': ['reheater'],
    'ECON': ['economizer', 'economiser'],
    'APH': ['air', 'preheater'],
    'ESP': ['electrostatic', 'precipitator'],
    'FD': ['forced', 'draft'],
    'FDF': ['forced', 'draft', 'fan'],
    'IDF': ['induced', 'draft', 'fan'],
    'COND': ['condenser'],
    'DEA': ['deaerator'],
    'BFP': ['boiler', 'feed', 'pump'],
    'CEP': ['condensate', 'extraction', 'pump'],
    'CWP': ['cooling', 'water', 'pump'],
    'CT': ['cooling', 'tower'],
    'CDU': ['crude', 'distillation', 'unit'],
    'VDU': ['vacuum', 'distillation', 'unit'],
    'FCC': ['fluid', 'catalytic', 'cracking'],
    'FCCU': ['fluid', 'catalytic', 'cracking', 'unit'],
    'HCU': ['hydrocracker', 'unit'],
    'HDT': ['hydrotreater'],
    'DHT': ['diesel', 'hydrotreater'],
    'NHT': ['naphtha', 'hydrotreater'],
    'CCR': ['continuous', 'catalytic', 'reformer'],
    'SRU': ['sulphur', 'recovery', 'unit'],
    'ARU': ['amine', 'recovery', 'unit'],
    'DHDS': ['diesel', 'hydro', 'desulphurization'],
    'OHCU': ['once', 'through', 'hydrocracker'],
    'VBU': ['visbreaker', 'unit'],
    'DCU': ['delayed', 'coker', 'unit'],
    'HGU': ['hydrogen', 'generation', 'unit'],
    'PSA': ['pressure', 'swing', 'adsorption'],

    # ===== EQUIPMENT — ROTATING =====
    'HP': ['high', 'pressure'],
    'LP': ['low', 'pressure'],
    'MP': ['medium', 'pressure'],
    'VHP': ['very', 'high', 'pressure'],
    'VFD': ['variable', 'frequency', 'drive'],
    'ASD': ['adjustable', 'speed', 'drive'],
    'RPM': ['revolutions', 'per', 'minute'],
    'BHP': ['brake', 'horsepower'],
    'NPSH': ['net', 'positive', 'suction', 'head'],

    # ===== ELECTRICAL =====
    'HV': ['high', 'voltage'],
    'LV': ['low', 'voltage'],
    'MV': ['medium', 'voltage'],
    'HT_ELEC': ['high', 'tension'],
    'LT': ['low', 'tension'],
    'MCC': ['motor', 'control', 'centre'],
    'PCC': ['power', 'control', 'centre'],
    'PMCC': ['power', 'motor', 'control', 'centre'],
    'SWG_ELEC': ['switchgear'],
    'ACB': ['air', 'circuit', 'breaker'],
    'VCB': ['vacuum', 'circuit', 'breaker'],
    'MCCB': ['moulded', 'case', 'circuit', 'breaker'],
    'ELCB': ['earth', 'leakage', 'circuit', 'breaker'],
    'RCCB': ['residual', 'current', 'circuit', 'breaker'],
    'DOL': ['direct', 'online', 'starter'],
    'ATS': ['automatic', 'transfer', 'switch'],
    'UPS': ['uninterruptible', 'power', 'supply'],
    'DG': ['diesel', 'generator'],
    'XFMR': ['transformer'],
    'CT_ELEC': ['current', 'transformer'],
    'PT_ELEC': ['potential', 'transformer'],
    'DB': ['distribution', 'board'],
    'PDB': ['power', 'distribution', 'board'],
    'LDB': ['lighting', 'distribution', 'board'],
    'SLD': ['single', 'line', 'diagram'],
    'GA_ELEC': ['general', 'arrangement', 'drawing'],
    'IP': ['ingress', 'protection'],
    'ATEX': ['explosive', 'atmosphere', 'hazardous', 'area'],
    'CACA': ['closed', 'air', 'circuit', 'cooled'],
    'TEFC': ['totally', 'enclosed', 'fan', 'cooled'],

    # ===== INSTRUMENTATION & CONTROL =====
    'PLC': ['programmable', 'logic', 'controller'],
    'DCS': ['distributed', 'control', 'system'],
    'SCADA': ['supervisory', 'control', 'data', 'acquisition'],
    'ESD': ['emergency', 'shutdown'],
    'SIS': ['safety', 'instrumented', 'system'],
    'SIL': ['safety', 'integrity', 'level'],
    'SIF': ['safety', 'instrumented', 'function'],
    'BMS': ['burner', 'management', 'system'],
    'FSSS': ['furnace', 'safeguard', 'supervisory', 'system'],
    'FGS': ['fire', 'gas', 'system'],
    'RTD': ['resistance', 'temperature', 'detector'],
    'TC_INST': ['thermocouple'],
    'TT': ['temperature', 'transmitter'],
    'PT_INST': ['pressure', 'transmitter'],
    'FT': ['flow', 'transmitter'],
    'LT_INST': ['level', 'transmitter'],
    'FI': ['flow', 'indicator'],
    'FIC': ['flow', 'indicator', 'controller'],
    'TI': ['temperature', 'indicator'],
    'TIC': ['temperature', 'indicator', 'controller'],
    'PI': ['pressure', 'indicator'],
    'PIC': ['pressure', 'indicator', 'controller'],
    'LI': ['level', 'indicator'],
    'LIC': ['level', 'indicator', 'controller'],
    'PDI': ['pressure', 'differential', 'indicator'],
    'AI': ['analog', 'input'],
    'AO': ['analog', 'output'],
    'DI_INST': ['digital', 'input'],
    'DO': ['digital', 'output'],
    'IO': ['input', 'output'],
    'HMI': ['human', 'machine', 'interface'],
    'HART': ['highway', 'addressable', 'remote', 'transducer'],
    'FF_INST': ['foundation', 'fieldbus'],
    'JB': ['junction', 'box'],
    'MJB': ['marshalling', 'junction', 'box'],
    'FJB': ['field', 'junction', 'box'],

    # ===== CIVIL & STRUCTURAL =====
    'RCC': ['reinforced', 'cement', 'concrete'],
    'PCC_CIVIL': ['plain', 'cement', 'concrete'],
    'PSC': ['prestressed', 'concrete'],
    'TMT': ['thermo', 'mechanical', 'treatment', 'rebar'],
    'HYSD': ['high', 'yield', 'strength', 'deformed'],
    'FGL': ['finished', 'ground', 'level'],
    'FFL': ['finished', 'floor', 'level'],
    'NGL': ['natural', 'ground', 'level'],
    'TOC': ['top', 'concrete'],
    'TOF': ['top', 'foundation'],
    'TOS': ['top', 'steel'],
    'BOF': ['bottom', 'foundation'],
    'GL': ['ground', 'level'],
    'EL': ['elevation', 'level'],
    'BOP': ['bottom', 'pipe'],
    'COP': ['centre', 'pipe'],
    'TOP': ['top', 'pipe'],
    'SBC': ['soil', 'bearing', 'capacity'],
    'SPT': ['standard', 'penetration', 'test'],
    'DCPT': ['dynamic', 'cone', 'penetration', 'test'],
    'PCC_PILING': ['precast', 'concrete', 'pile'],
    'WLT': ['working', 'load', 'test'],
    'PLT': ['pile', 'load', 'test'],

    # ===== FIRE & SAFETY =====
    'HSE': ['health', 'safety', 'environment'],
    'EHS': ['environment', 'health', 'safety'],
    'HSSE': ['health', 'safety', 'security', 'environment'],
    'HAZID': ['hazard', 'identification'],
    'HAZOP': ['hazard', 'operability', 'study'],
    'SRA': ['safety', 'risk', 'assessment'],
    'QRA': ['quantitative', 'risk', 'assessment'],
    'FERA': ['fire', 'explosion', 'risk', 'assessment'],
    'PPE': ['personal', 'protective', 'equipment'],
    'PTW': ['permit', 'work'],
    'SIMOPS': ['simultaneous', 'operations'],
    'JSA': ['job', 'safety', 'analysis'],
    'TBT': ['tool', 'box', 'talk'],
    'MSDS': ['material', 'safety', 'data', 'sheet'],
    'SDS': ['safety', 'data', 'sheet'],
    'FPS': ['fire', 'protection', 'system'],
    'FHS': ['fire', 'hydrant', 'system'],
    'FAS': ['fire', 'alarm', 'system'],
    'VESDA': ['very', 'early', 'smoke', 'detection'],
    'DCP': ['dry', 'chemical', 'powder'],
    'MV_FIRE': ['medium', 'velocity', 'spray'],
    'HV_FIRE': ['high', 'velocity', 'spray'],
    'AFFF': ['aqueous', 'film', 'forming', 'foam'],
    'HVAC': ['heating', 'ventilation', 'air', 'conditioning'],

    # ===== PROCESS ENGINEERING =====
    'FEED': ['front', 'end', 'engineering', 'design'],
    'BED': ['basic', 'engineering', 'design'],
    'DED': ['detailed', 'engineering', 'design'],
    'BEDP': ['basic', 'engineering', 'design', 'package'],
    'HMB': ['heat', 'material', 'balance'],
    'UFD': ['utility', 'flow', 'diagram'],
    'ELD': ['electrical', 'load', 'diagram'],
    'CSD': ['cause', 'effect', 'shutdown', 'diagram'],
    'TEMA': ['tubular', 'exchanger', 'manufacturers', 'association'],
    'API': ['american', 'petroleum', 'institute'],
    'LMTD': ['log', 'mean', 'temperature', 'difference'],
    'COP_PROC': ['coefficient', 'performance'],
    'MAWP': ['maximum', 'allowable', 'working', 'pressure'],
    'MAOT': ['maximum', 'allowable', 'operating', 'temperature'],
    'MAP': ['maximum', 'allowable', 'pressure'],
    'MWP': ['maximum', 'working', 'pressure'],
    'DP_PROC': ['design', 'pressure'],
    'DT': ['design', 'temperature'],
    'OP': ['operating', 'pressure'],
    'OT': ['operating', 'temperature'],
    'CA': ['corrosion', 'allowance'],
    'BP': ['boiling', 'point'],
    'FP': ['flash', 'point'],
    'LEL': ['lower', 'explosive', 'limit'],
    'UEL': ['upper', 'explosive', 'limit'],
    'LFL': ['lower', 'flammable', 'limit'],
    'UFL': ['upper', 'flammable', 'limit'],
    'NPSHA': ['net', 'positive', 'suction', 'head', 'available'],
    'NPSHR': ['net', 'positive', 'suction', 'head', 'required'],

    # ===== BOILER & POWER PLANT =====
    'OCB': ['oil', 'circuit', 'breaker'],
    'OCO': ['old', 'crude', 'oil'],
    'SH_BOILER': ['superheater'],
    'RH_BOILER': ['reheater'],
    'ECON_BOILER': ['economizer'],
    'HPSH': ['high', 'pressure', 'superheater'],
    'LPSH': ['low', 'pressure', 'superheater'],
    'IPMH': ['intermediate', 'pressure', 'mixed', 'header'],
    'HPH': ['high', 'pressure', 'heater'],
    'LPH': ['low', 'pressure', 'heater'],
    'CPH': ['condensate', 'polishing', 'heater'],
    'PRDS': ['pressure', 'reducing', 'desuperheating'],
    'CBD': ['continuous', 'blowdown'],
    'IBD': ['intermittent', 'blowdown'],
    'FWH': ['feed', 'water', 'heater'],
    'DM': ['demineralized', 'water'],
    'DMW': ['demineralized', 'water'],
    'DMP': ['demineralized', 'water', 'plant'],
    'CW': ['cooling', 'water'],
    'RW': ['raw', 'water'],
    'SW_WATER': ['service', 'water'],
    'IA': ['instrument', 'air'],
    'PA': ['plant', 'air'],
    'SA_AIR': ['service', 'air'],
    'NG': ['natural', 'gas'],
    'FO_FUEL': ['fuel', 'oil'],
    'FG': ['fuel', 'gas'],
    'LDO': ['light', 'diesel', 'oil'],
    'HSD': ['high', 'speed', 'diesel'],
    'HFO': ['heavy', 'fuel', 'oil'],
    'LSHS': ['low', 'sulphur', 'heavy', 'stock'],
    'LPG': ['liquefied', 'petroleum', 'gas'],
    'LNG': ['liquefied', 'natural', 'gas'],
    'PNG': ['piped', 'natural', 'gas'],

    # ===== INDIAN CODES & STANDARDS BODIES =====
    'IBR': ['indian', 'boiler', 'regulations'],
    'BIS': ['bureau', 'indian', 'standards'],
    'IS': ['indian', 'standard'],
    'OISD': ['oil', 'industry', 'safety', 'directorate'],
    'PESO': ['petroleum', 'explosives', 'safety', 'organisation'],
    'PNGRB': ['petroleum', 'natural', 'gas', 'regulatory', 'board'],
    'CEA': ['central', 'electricity', 'authority'],
    'AERB': ['atomic', 'energy', 'regulatory', 'board'],
    'CPCB': ['central', 'pollution', 'control', 'board'],
    'SPCB': ['state', 'pollution', 'control', 'board'],
    'MoEFCC': ['ministry', 'environment', 'forest', 'climate', 'change'],
    'NBC': ['national', 'building', 'code'],
    'IRC': ['indian', 'roads', 'congress'],
    'CPWD': ['central', 'public', 'works', 'department'],

    # ===== INTERNATIONAL CODES & STANDARDS =====
    'ASME': ['american', 'society', 'mechanical', 'engineers'],
    'ASTM': ['american', 'society', 'testing', 'materials'],
    'AWS': ['american', 'welding', 'society'],
    'ANSI': ['american', 'national', 'standards', 'institute'],
    'NFPA': ['national', 'fire', 'protection', 'association'],
    'IEC': ['international', 'electrotechnical', 'commission'],
    'IEEE': ['institute', 'electrical', 'electronics', 'engineers'],
    'ISA': ['instrument', 'society', 'america'],
    'NEMA': ['national', 'electrical', 'manufacturers', 'association'],
    'OSHA': ['occupational', 'safety', 'health'],
    'PED': ['pressure', 'equipment', 'directive'],

    # ===== INDIAN PSUs & ORGANISATIONS =====
    'HPCL': ['hindustan', 'petroleum'],
    'BPCL': ['bharat', 'petroleum'],
    'IOCL': ['indian', 'oil'],
    'ONGC': ['oil', 'natural', 'gas', 'corporation'],
    'GAIL': ['gas', 'authority', 'india'],
    'NTPC': ['national', 'thermal', 'power', 'corporation'],
    'BHEL': ['bharat', 'heavy', 'electricals'],
    'EIL': ['engineers', 'india', 'limited'],
    'MECON': ['metallurgical', 'engineering', 'consultants'],
    'PDIL': ['projects', 'development', 'india'],
    'CPCL': ['chennai', 'petroleum', 'corporation'],
    'MRPL': ['mangalore', 'refinery', 'petrochemicals'],
    'HMEL': ['hpcl', 'mittal', 'energy'],
    'NRL': ['numaligarh', 'refinery'],
    'BRPL': ['bongaigaon', 'refinery'],

    # ===== TENDER / CONTRACT / COMMERCIAL =====
    'PRS': ['price', 'reduction', 'schedule'],
    'LD': ['liquidated', 'damages'],
    'BOQ': ['bill', 'quantities'],
    'SOR': ['schedule', 'rates'],
    'EMD': ['earnest', 'money', 'deposit'],
    'SD': ['security', 'deposit'],
    'BG': ['bank', 'guarantee'],
    'PBG': ['performance', 'bank', 'guarantee'],
    'ABG': ['advance', 'bank', 'guarantee'],
    'GCC': ['general', 'conditions', 'contract'],
    'SCC': ['special', 'conditions', 'contract'],
    'NIT': ['notice', 'inviting', 'tender'],
    'DLP': ['defect', 'liability', 'period'],
    'LSTK': ['lump', 'sum', 'turnkey'],
    'PMC': ['project', 'management', 'consultant'],
    'EPC': ['engineering', 'procurement', 'construction'],
    'EPCM': ['engineering', 'procurement', 'construction', 'management'],
    'ITB': ['invitation', 'bid'],
    'LOI': ['letter', 'intent'],
    'LOA': ['letter', 'acceptance'],
    'WO': ['work', 'order'],
    'PO': ['purchase', 'order'],
    'GRN': ['goods', 'receipt', 'note'],
    'RA': ['running', 'account', 'bill'],
    'RA_BILL': ['running', 'account', 'bill'],
    'MB': ['measurement', 'book'],
    'SOW': ['scope', 'work'],
    'TOR': ['terms', 'reference'],
    'RFP': ['request', 'proposal'],
    'RFQ': ['request', 'quotation'],
    'EOI': ['expression', 'interest'],
    'TBE': ['techno', 'commercial', 'bid', 'evaluation'],
    'CBE': ['commercial', 'bid', 'evaluation'],
    'PAC': ['project', 'acceptance', 'certificate'],
    'FAC': ['final', 'acceptance', 'certificate'],
    'TOC_COMM': ['taking', 'over', 'certificate'],
    'CCPU': ['contract', 'close', 'pending', 'utilization'],
    'CIF': ['cost', 'insurance', 'freight'],
    'FOB': ['free', 'on', 'board'],
    'FOR': ['free', 'on', 'rail'],
    'FOT': ['free', 'on', 'truck'],
    'EXW': ['ex', 'works'],
    'DDP': ['delivered', 'duty', 'paid'],
    'LC': ['letter', 'credit'],
    'GST': ['goods', 'services', 'tax'],
    'SGST': ['state', 'goods', 'services', 'tax'],
    'CGST': ['central', 'goods', 'services', 'tax'],
    'IGST': ['integrated', 'goods', 'services', 'tax'],
    'TDS': ['tax', 'deducted', 'source'],
    'WCT': ['works', 'contract', 'tax'],
    'PAN': ['permanent', 'account', 'number'],
    'GSTIN': ['goods', 'services', 'tax', 'identification', 'number'],
    'MSME': ['micro', 'small', 'medium', 'enterprises'],
    'SSI': ['small', 'scale', 'industry'],
    'OEM': ['original', 'equipment', 'manufacturer'],
    'AMC': ['annual', 'maintenance', 'contract'],
    'CAMC': ['comprehensive', 'annual', 'maintenance', 'contract'],
    'SLA': ['service', 'level', 'agreement'],
    'MOA': ['memorandum', 'agreement'],
    'MOU': ['memorandum', 'understanding'],
    'NDA': ['non', 'disclosure', 'agreement'],
    'CPM': ['critical', 'path', 'method'],
    'PERT': ['program', 'evaluation', 'review', 'technique'],
    'WBS': ['work', 'breakdown', 'structure'],
    'FIDIC': ['international', 'federation', 'consulting', 'engineers'],

    # ===== PROJECT MANAGEMENT =====
    'PMB': ['project', 'management', 'baseline'],
    'PMS': ['project', 'management', 'system'],
    'PMT': ['project', 'management', 'team'],
    'PGMA': ['project', 'general', 'management', 'agency'],
    'OBE': ['overcome', 'by', 'events'],
    'DBE': ['design', 'basis', 'event'],
    'RLOC': ['revised', 'letter', 'credit'],
    'VO': ['variation', 'order'],
    'EOT': ['extension', 'time'],
    'COD': ['commercial', 'operation', 'date'],
    'MC': ['mechanical', 'completion'],
    'RFS': ['ready', 'start', 'up'],
    'RFSU': ['ready', 'start', 'up'],
    'OSBL': ['outside', 'battery', 'limits'],
    'ISBL': ['inside', 'battery', 'limits'],
    'BL': ['battery', 'limits'],

    # ===== CONSTRUCTION & SITE =====
    'TA': ['turnaround'],
    'MDMT': ['minimum', 'design', 'metal', 'temperature'],
    'PSSR': ['pre', 'startup', 'safety', 'review'],
    'PHA': ['process', 'hazard', 'analysis'],
    'OFC': ['optical', 'fibre', 'cable'],
    'UG': ['underground'],
    'AG': ['above', 'ground'],
    'OHL': ['overhead', 'line'],
    'RHS': ['rectangular', 'hollow', 'section'],
    'CHS': ['circular', 'hollow', 'section'],
    'SHS': ['square', 'hollow', 'section'],
    'PE': ['polyethylene'],
    'PVC': ['polyvinyl', 'chloride'],
    'CPVC': ['chlorinated', 'polyvinyl', 'chloride'],
    'GRE': ['glass', 'reinforced', 'epoxy'],
    'SS_STRUCT': ['structural', 'steel'],
    'RC': ['reinforced', 'concrete'],
    'PC': ['precast', 'concrete'],
    'SBR': ['styrene', 'butadiene', 'rubber'],
    'EPDM': ['ethylene', 'propylene', 'rubber'],
    'NBR': ['nitrile', 'butadiene', 'rubber'],

    # ===== PAINTING & COATINGS =====
    'DFT': ['dry', 'film', 'thickness'],
    'WFT': ['wet', 'film', 'thickness'],
    'SSPC': ['society', 'protective', 'coatings'],
    'SIS_PAINT': ['swedish', 'standard', 'surface', 'preparation'],

    # ===== REFRACTORY =====
    'CC': ['castable', 'concrete'],
    'LWC': ['light', 'weight', 'castable'],
    'DRHC': ['dense', 'refractory', 'castable'],
    'HAS_REF': ['high', 'alumina', 'castable'],
    'CF': ['ceramic', 'fibre'],
    'CFB': ['ceramic', 'fibre', 'blanket'],
    'CFM': ['ceramic', 'fibre', 'module'],
    'IFB': ['insulating', 'fire', 'brick'],
    'HDB': ['high', 'duty', 'brick'],
    'SHD': ['super', 'high', 'duty'],

    # ===== CATHODIC PROTECTION & CORROSION =====
    'CP': ['cathodic', 'protection'],
    'ICCP': ['impressed', 'current', 'cathodic', 'protection'],
    'SACP': ['sacrificial', 'anode', 'cathodic', 'protection'],
    'CUI': ['corrosion', 'under', 'insulation'],
    'SCC_CORR': ['stress', 'corrosion', 'cracking'],
    'HIC': ['hydrogen', 'induced', 'cracking'],
    'SOHIC': ['stress', 'oriented', 'hydrogen', 'induced', 'cracking'],
    'SSC': ['sulphide', 'stress', 'cracking'],
    'MIC': ['microbiologically', 'influenced', 'corrosion'],

    # ===== INSULATION & CLADDING =====
    'MW': ['mineral', 'wool'],
    'RW': ['rock', 'wool'],
    'CCS': ['calcium', 'silicate'],
    'PUF': ['polyurethane', 'foam'],
    'PIR': ['polyisocyanurate'],
    'XPS': ['extruded', 'polystyrene'],
    'EPS': ['expanded', 'polystyrene'],
    'AL': ['aluminium', 'cladding'],
}


def expand_abbreviations(query: str) -> list[str]:
    """Return extra search terms by expanding abbreviations found in the query."""
    words = query.strip().split()
    extra = []
    for w in words:
        key = w.upper().strip('"\'.,;:')
        if key in ABBREVIATIONS:
            extra.extend(ABBREVIATIONS[key])
    return extra


def _fts_search(conn, fts_query, doc_ids, doc_id=None, page_type=None, limit=25):
    """Execute an FTS5 query and return result rows (may raise OperationalError)."""
    conds = ["pages_fts MATCH ?"]
    params: list = [fts_query]

    ph = ','.join('?' * len(doc_ids))
    conds.append(f"p.doc_id IN ({ph})")
    params.extend(doc_ids)

    if doc_id is not None:
        conds.append("p.doc_id = ?")
        params.append(doc_id)
    if page_type:
        conds.append("p.page_type = ?")
        params.append(page_type)

    params.append(limit)

    return conn.execute(f"""
        SELECT p.doc_id, d.title as doc_title, d.total_pages,
               p.page_num, p.breadcrumb, p.page_type, p.char_count,
               snippet(pages_fts, 0, '>>>', '<<<', '...', 40) as snippet, rank
        FROM pages_fts
        JOIN pages p ON p.id = pages_fts.rowid
        JOIN documents d ON d.id = p.doc_id
        WHERE {' AND '.join(conds)}
        ORDER BY rank LIMIT ?
    """, params).fetchall()


def project_doc_ids(conn, project: str) -> list[int]:
    return [r['id'] for r in conn.execute(
        "SELECT id FROM documents WHERE project = ? OR project LIKE ?",
        (project, f"{project}/%")
    ).fetchall()]


def all_known_folders(conn) -> list[str]:
    folders = set()
    for table in ("documents", "chat_threads", "ingestion_jobs"):
        try:
            rows = conn.execute(f"SELECT DISTINCT project FROM {table}").fetchall()
        except sqlite3.OperationalError:
            rows = []
        for row in rows:
            value = (row["project"] or "").strip().strip("/")
            if not value:
                continue
            parts = value.split("/")
            for i in range(1, len(parts) + 1):
                folders.add("/".join(parts[:i]))
    return sorted(folders)


def _folder_entry(folder: str) -> dict:
    return {
        "project": folder,
        "folder": folder,
        "display_name": folder.rsplit("/", 1)[-1],
        "parent": folder.rsplit("/", 1)[0] if "/" in folder else None,
        "depth": folder.count("/"),
        "kind": "folder",
    }


def _document_entry(conn, row) -> dict:
    meta = json.loads(row["metadata"]) if row["metadata"] else {}
    sec_count = conn.execute(
        "SELECT COUNT(*) FROM sections WHERE doc_id = ?",
        (row["id"],)
    ).fetchone()[0]
    return {
        "id": row["id"],
        "project": row["project"],
        "folder": row["project"],
        "title": row["title"],
        "filename": row["filename"],
        "total_pages": row["total_pages"],
        "sections": sec_count,
        "document_type": meta.get("document_type"),
        "kind": "file",
    }


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

SSE_PORT = 8200

mcp = FastMCP(
    "Document RAG",
    instructions="""\
Search and navigate engineering folder documents (specifications, procedures, QAPs, tender docs).
All searches are scoped by folder. A folder search includes all of its sub-folders. Use list_folders first.
Use list_folder_entries when you need folders/files within a specific folder.

=== SEARCH STRATEGY ===

You have two search tools — use them together:

1. search_pages — FTS keyword search (fast, precise when you know the words)
   - Automatically expands abbreviations (PRS → price reduction schedule, MOC → material construction, etc.)
   - Uses AND first for precision, then falls back to OR for recall
   - Special characters like "3.2" are handled safely
   - Still benefits from trying 2-4 query variations with different keywords

2. semantic_search — Vector/embedding search (finds conceptually similar content)
   - Use when you don't know the exact terminology in the documents
   - Accepts natural language questions: "what are the penalty clauses for late delivery?"
   - Finds content by meaning, not just exact words
   - Slower than FTS but much better recall for vague queries

3. get_section — Read entire section content in one call
   - After finding a section via search_sections, use get_section(doc_id, heading) to
     read all pages at once instead of paging through manually

RECOMMENDED WORKFLOW:
1. Start with search_pages using the most obvious keywords (2-3 variations)
2. If FTS returns too few results, use semantic_search with a natural language query
3. Use search_sections to find which section/document likely has the answer
4. Use get_section to read the full section content in one call
5. Use get_page with include_adjacent=true for surrounding context

TIPS:
- Strip question words: "What is the MOC of superheater coil tubes?" → "superheater coil tube material"
- Try synonyms: tube ↔ pipe ↔ coil | material ↔ grade ↔ alloy ↔ specification
- Abbreviations are auto-expanded, but still try both forms for best coverage
- Look for ANNEXURE references — specs often say "as per Annexure X"
- Use search_sections to find which DOCUMENT likely has the answer, then search within it

After finding relevant pages, use get_page with include_adjacent=true to see surrounding context.
If Marker output looks garbled, use reextract_page or reextract_table for a fresh PDF extraction.

=== DATA CORRECTION (YOU CAN AND SHOULD FIX WHAT YOU SEE) ===

*** CRITICAL RULE: NEVER ALTER DOCUMENT CONTENT ***
The page content is extracted from official engineering documents — specifications,
contracts, procedures, QAPs. This content is SACRED. Even if you see a spelling mistake,
a grammatical error, or a wrong number in the document text, DO NOT CHANGE IT. That is
what the original document says, and it must remain exactly as-is. These are legal and
contractual documents — altering their content would be falsification.

What you CAN correct (extraction artifacts only):
  ✓ Heading structure — levels, missing/false headings (the EXTRACTOR got these wrong, not the document)
  ✓ OCR artifacts — garbled characters from scanning (e.g. "tbe" that is clearly "the" due to
    a scanning error, random symbols injected by OCR). Only fix text that is OBVIOUSLY a scanning
    artifact, never "correct" what might be the original document's actual text.
  ✓ Document structure — splits, merges, page classification, breadcrumbs
  ✓ Metadata — titles, types, revision numbers, keywords (these are YOUR labels, not document content)
  ✓ Running headers/footers — repeated extraction artifacts cluttering every page
  ✓ Quality flags — flagging problems for human review
  ✓ Cross-references and keywords — adding search aids

What you must NEVER do:
  ✗ Fix spelling or grammar in document content — that's what the document actually says
  ✗ "Correct" numbers, dates, or technical values — even if they look wrong
  ✗ Rewrite or rephrase any document text
  ✗ Remove content that looks redundant — it may be intentional
  When in doubt, leave the content alone and flag_low_quality instead.

Your corrections improve the system permanently: they write to both the live database
(immediate effect on search results) AND a sidecar JSON file next to the PDF. When the
document is re-ingested later, your corrections are automatically replayed. The system
gets better every time you use it.

WHEN TO CORRECT (do this as you go, not as a separate task):
- You see a heading that shouldn't be one (e.g. a table row detected as H2):
  → remove_heading(doc_id, page_num, text_prefix)
- A real heading was missed by the extractor:
  → add_heading(doc_id, page_num, text, level)
- Heading is at the wrong level (H3 should be H1):
  → change_heading_level(doc_id, page_num, text_prefix, new_level)
- Heading text is garbled from OCR (not a document correction — an extraction fix):
  → rename_heading(doc_id, page_num, old_text_prefix, new_text)
- Two documents should actually be one (were incorrectly split):
  → merge_documents(doc_id_a, doc_id_b)
- One document contains multiple logical documents:
  → split_document(doc_id, at_page)
- OCR artifact in content (garbled characters from scanning, NOT a document typo):
  → fix_ocr_text(doc_id, page_num, old_text, new_text)
- A page is classified wrong (text marked as drawing, or vice versa):
  → reclassify_page(doc_id, page_num, new_type)
- A page is junk (blank, duplicate, garbage OCR):
  → skip_page(doc_id, page_num)
- A page belongs in a different document:
  → move_page_to_document(page_num, from_doc_id, to_doc_id)
- The breadcrumb/context on a page is wrong:
  → set_page_breadcrumb(doc_id, page_num, breadcrumb)
- A repeated header/footer clutters the content (extraction artifact on every page):
  → add_running_header(doc_id, text) — strips it from all pages immediately
- Document title is wrong or unhelpful:
  → set_document_title(doc_id, title)
- You can identify the document type, number, or revision:
  → set_document_type, set_document_number, set_revision
- You spot cross-references between documents:
  → add_cross_reference(doc_id, page_num, target_doc_id, context)
  → link_documents(doc_id, related_doc_id, relationship)
- You want to tag keywords or equipment for better future search:
  → add_keywords(doc_id, keywords_csv)
  → add_equipment_tags(doc_id, tags_csv)
- A page has terrible OCR quality or suspicious content:
  → flag_low_quality(doc_id, page_num, reason) — flag for human review, don't alter
  → suggest_reocr(doc_id, reason)
- Two documents look like duplicates:
  → flag_duplicate(doc_id, duplicate_of_doc_id)

Be proactive with STRUCTURAL fixes. If the document title is "f 41  maintenance work
procedure   s" and you can see from the content it's actually "F-41: Maintenance Work
Procedure - Scaffolding", fix the title. If a page has 50 false headings from table rows,
remove them. But never touch the actual document text.
""",
    port=SSE_PORT,
)

register_correction_tools(mcp, get_db)


# ===== Discovery =====

@mcp.tool()
def list_folders() -> ToolResult:
    """List top-level folders with aggregate document and page counts."""
    with get_db() as conn:
        exact_stats = {
            row["project"]: {
                "documents": row["docs"],
                "total_pages": row["pages"] or 0,
            }
            for row in conn.execute("""
                SELECT project, COUNT(*) as docs, SUM(total_pages) as pages
                FROM documents GROUP BY project ORDER BY project
            """).fetchall()
        }

        folders = [folder for folder in all_known_folders(conn) if "/" not in folder]
        results = []
        for folder in folders:
            prefix = folder + "/"
            docs = 0
            pages = 0
            for candidate, stats in exact_stats.items():
                if candidate == folder or candidate.startswith(prefix):
                    docs += stats["documents"]
                    pages += stats["total_pages"]
            results.append({
                "project": folder,
                "folder": folder,
                "display_name": folder.rsplit("/", 1)[-1],
                "parent": folder.rsplit("/", 1)[0] if "/" in folder else None,
                "depth": folder.count("/"),
                "documents": docs,
                "total_pages": pages,
            })
        return results


@mcp.tool()
def list_folder_entries(
    project: str,
    include: str = "both",
    recursive: bool = False,
) -> ToolResult:
    """List folders/files within a folder.

    Args:
        project: Folder path to inspect.
        include: 'folders', 'files', or 'both'.
        recursive: If true, include all descendants. If false, only direct children/direct files.
    """
    project = (project or "").strip().strip("/")
    if not project:
        return {"error": "project is required"}
    if include not in {"folders", "files", "both"}:
        return {"error": "include must be one of: folders, files, both"}

    with get_db() as conn:
        folder_items = []
        file_items = []
        prefix = project + "/"
        if include in {"folders", "both"}:
            for folder in all_known_folders(conn):
                if not folder.startswith(prefix):
                    continue
                suffix = folder[len(prefix):]
                if not recursive and "/" in suffix:
                    continue
                folder_items.append(_folder_entry(folder))

        if include in {"files", "both"}:
            doc_rows = conn.execute(
                "SELECT * FROM documents WHERE project = ? OR project LIKE ? ORDER BY project, id",
                (project, f"{project}/%")
            ).fetchall()
            for row in doc_rows:
                relative_folder = row["project"][len(project):].strip("/")
                if not recursive and relative_folder:
                    continue
                file_items.append(_document_entry(conn, row))

        return {
            "project": project,
            "folder": project,
            "include": include,
            "recursive": recursive,
            "folder_count": len(folder_items),
            "file_count": len(file_items),
            "entries": folder_items + file_items,
            "folders": folder_items,
            "files": file_items,
        }
@mcp.tool()
def list_documents(project: str) -> ToolResult:
    """List all documents in a folder scope.

    Args:
        project: Folder path (from list_folders). Includes sub-folders.
    """
    with get_db() as conn:
        docs = conn.execute(
            "SELECT * FROM documents WHERE project = ? OR project LIKE ? ORDER BY project, id",
            (project, f"{project}/%")
        ).fetchall()

        results = []
        for d in docs:
            meta = json.loads(d["metadata"]) if d["metadata"] else {}
            sec_count = conn.execute(
                "SELECT COUNT(*) FROM sections WHERE doc_id = ?", (d["id"],)
            ).fetchone()[0]

            results.append({
                "id": d["id"],
                "title": d["title"],
                "filename": d["filename"],
                "total_pages": d["total_pages"],
                "sections": sec_count,
                "document_type": meta.get("document_type"),
                "prepared_by": meta.get("prepared_by"),
                "summary": meta.get("summary"),
            })

        return results


@mcp.tool()
def get_document_info(doc_id: int) -> ToolResult:
    """Get full metadata for a document including LLM-extracted details.

    Args:
        doc_id: Document ID.
    """
    with get_db() as conn:
        d = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not d:
            return {"error": f"Document {doc_id} not found"}

        meta = json.loads(d["metadata"]) if d["metadata"] else {}

        types = {}
        for r in conn.execute(
            "SELECT page_type, COUNT(*) as n FROM pages WHERE doc_id = ? GROUP BY page_type",
            (doc_id,)
        ):
            types[r["page_type"]] = r["n"]

        return {
            "id": d["id"], "project": d["project"], "title": d["title"],
            "filename": d["filename"], "pdf_path": d["pdf_path"],
            "total_pages": d["total_pages"], "page_types": types,
            "document_number": meta.get("document_number"),
            "revision": meta.get("revision"),
            "date": meta.get("date"),
            "prepared_by": meta.get("prepared_by"),
            "prepared_for": meta.get("prepared_for"),
            "project_name": meta.get("project_name"),
            "document_type": meta.get("document_type"),
            "summary": meta.get("summary"),
            "equipment_tags": meta.get("equipment_tags"),
            "applicable_codes": meta.get("applicable_codes"),
            "keywords": meta.get("keywords"),
        }


@mcp.tool()
def get_toc(doc_id: int, max_level: int = 4) -> ToolResult:
    """Get the section tree (table of contents) for a document.

    Args:
        doc_id: Document ID.
        max_level: Maximum heading depth (default 4).
    """
    with get_db() as conn:
        d = conn.execute("SELECT title FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not d:
            return {"error": f"Document {doc_id} not found"}

        sections = conn.execute(
            """SELECT heading, level, page_start, page_end, breadcrumb
               FROM sections WHERE doc_id = ? AND level <= ? ORDER BY seq""",
            (doc_id, max_level)
        ).fetchall()

        return {
            "doc_title": d["title"],
            "sections": [
                {"heading": s["heading"], "level": s["level"],
                 "pages": f"{s['page_start']}-{s['page_end']}",
                 "breadcrumb": s["breadcrumb"]}
                for s in sections
            ]
        }


# ===== Search =====

@mcp.tool()
def search_pages(
    project: str,
    query: str,
    doc_id: Optional[int] = None,
    page_type: Optional[str] = None,
    max_results: int = 10
) -> ToolResult:
    """Full-text keyword search across a folder's pages.

    Automatically expands known abbreviations (e.g. PRS → price reduction schedule)
    and falls back to OR matching for better recall when exact AND matching is too strict.

    Args:
        project: Folder path (required). Includes sub-folders.
        query: 2-5 keyword terms (not a full sentence). Drop stop words.
               Examples: "superheater tube material", "SA 213 grade", "PRS clause"
        doc_id: Optional. Restrict to one document.
        page_type: Optional. Filter: 'text', 'drawing', 'table', 'toc', 'cover'.
        max_results: Default 10, max 25.

    Returns:
        Matching pages with doc_title, page_num, breadcrumb, snippet, and rank.
    """
    clean = sanitize_fts(query)
    if not clean:
        return {"error": "Empty query"}

    cap = min(max_results, 25)

    with get_db() as conn:
        ids = project_doc_ids(conn, project)
        if not ids:
            return {"error": f"No documents in folder '{project}'"}

        results = []
        seen = set()

        def _collect(rows):
            for r in rows:
                key = (r["doc_id"], r["page_num"])
                if key not in seen:
                    seen.add(key)
                    results.append(r)

        # --- Pass 1: AND with original terms (highest precision) ---
        try:
            _collect(_fts_search(conn, clean, ids, doc_id, page_type, cap))
        except sqlite3.OperationalError:
            pass

        # --- Pass 2: AND with abbreviation-expanded terms ---
        extra = expand_abbreviations(query)
        if extra and len(results) < cap:
            expanded = clean + ' ' + sanitize_fts(' '.join(extra))
            try:
                _collect(_fts_search(conn, expanded, ids, doc_id, page_type, cap))
            except sqlite3.OperationalError:
                pass

        # --- Pass 3: OR across all terms for recall ---
        if len(results) < cap:
            all_terms = clean.split()
            if extra:
                all_terms += sanitize_fts(' '.join(extra)).split()
            # Deduplicate while preserving order
            seen_t = set()
            unique = []
            for t in all_terms:
                tl = t.lower().strip('"')
                if tl and tl not in seen_t:
                    seen_t.add(tl)
                    unique.append(t)
            if len(unique) > 1:
                or_query = ' OR '.join(unique)
                try:
                    _collect(_fts_search(conn, or_query, ids, doc_id, page_type, cap))
                except sqlite3.OperationalError:
                    pass

        final = results[:cap]
        return {
            "query": query,
            "sanitized": clean,
            "expanded_terms": extra if extra else None,
            "result_count": len(final),
            "results": [
                {"doc_id": r["doc_id"], "doc_title": r["doc_title"],
                 "page_num": r["page_num"], "total_pages": r["total_pages"],
                 "breadcrumb": r["breadcrumb"], "page_type": r["page_type"],
                 "snippet": r["snippet"], "rank": round(r["rank"], 4)}
                for r in final
            ]
        }


@mcp.tool()
def search_sections(project: str, query: str, doc_id: Optional[int] = None) -> ToolResult:
    """Search section headings within a folder scope.

    Args:
        project: Folder path. Includes sub-folders.
        query: Keywords to match in section headings.
        doc_id: Optional. Restrict to one document.
    """
    with get_db() as conn:
        conds = ["(s.heading LIKE ? OR s.breadcrumb LIKE ?)", "(d.project = ? OR d.project LIKE ?)"]
        params = [f"%{query}%", f"%{query}%", project, f"{project}/%"]

        if doc_id is not None:
            conds.append("s.doc_id = ?")
            params.append(doc_id)

        rows = conn.execute(f"""
            SELECT s.doc_id, d.title as doc_title, s.heading, s.level,
                   s.breadcrumb, s.page_start, s.page_end
            FROM sections s JOIN documents d ON d.id = s.doc_id
            WHERE {' AND '.join(conds)}
            ORDER BY s.doc_id, s.seq
        """, params).fetchall()

        return {
            "query": query, "result_count": len(rows),
            "results": [dict(r) for r in rows]
        }


@mcp.tool()
def get_section(doc_id: int, heading: str, max_pages: int = 20) -> ToolResult:
    """Get all pages belonging to a section, identified by heading text.

    Use this after search_sections to read the full content of a section
    without having to page through manually.

    Args:
        doc_id: Document ID.
        heading: Section heading text (partial match — use a distinctive substring).
        max_pages: Maximum pages to return (default 20, max 30).
    """
    with get_db() as conn:
        d = conn.execute("SELECT title FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not d:
            return {"error": f"Document {doc_id} not found"}

        section = conn.execute(
            "SELECT * FROM sections WHERE doc_id = ? AND heading LIKE ? ORDER BY seq LIMIT 1",
            (doc_id, f"%{heading}%")
        ).fetchone()

        if not section:
            # Try case-insensitive
            section = conn.execute(
                "SELECT * FROM sections WHERE doc_id = ? AND LOWER(heading) LIKE LOWER(?) ORDER BY seq LIMIT 1",
                (doc_id, f"%{heading}%")
            ).fetchone()

        if not section:
            return {"error": f"No section matching '{heading}' in document {doc_id}"}

        page_cap = min(max_pages, 30)
        pages = conn.execute(
            "SELECT page_num, content, breadcrumb, page_type FROM pages "
            "WHERE doc_id = ? AND page_num BETWEEN ? AND ? ORDER BY page_num LIMIT ?",
            (doc_id, section['page_start'], section['page_end'], page_cap)
        ).fetchall()

        return {
            "doc_id": doc_id,
            "doc_title": d["title"],
            "section": {
                "heading": section["heading"],
                "level": section["level"],
                "breadcrumb": section["breadcrumb"],
                "pages": f"{section['page_start']}-{section['page_end']}"
            },
            "page_count": len(pages),
            "truncated": len(pages) == page_cap,
            "pages": [
                {"page_num": p["page_num"], "breadcrumb": p["breadcrumb"],
                 "page_type": p["page_type"], "content": p["content"]}
                for p in pages
            ]
        }


# ===== Navigation =====

@mcp.tool()
def get_page(doc_id: int, page_num: int, include_adjacent: bool = False) -> ToolResult:
    """Get full content of a specific page with section context.

    Args:
        doc_id: Document ID (from search results).
        page_num: Page number.
        include_adjacent: If True, also returns prev/next page content.
    """
    with get_db() as conn:
        d = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not d:
            return {"error": f"Document {doc_id} not found"}

        p = conn.execute(
            "SELECT * FROM pages WHERE doc_id = ? AND page_num = ?", (doc_id, page_num)
        ).fetchone()
        if not p:
            return {"error": f"Page {page_num} not found", "total_pages": d["total_pages"]}

        result = {
            "doc_id": doc_id, "doc_title": d["title"], "project": d["project"],
            "page_num": p["page_num"], "total_pages": d["total_pages"],
            "breadcrumb": p["breadcrumb"], "page_type": p["page_type"],
            "content": p["content"]
        }

        if include_adjacent:
            for direction, offset in [("prev_page", -1), ("next_page", 1)]:
                adj = conn.execute(
                    "SELECT page_num, content, breadcrumb, page_type FROM pages WHERE doc_id = ? AND page_num = ?",
                    (doc_id, page_num + offset)
                ).fetchone()
                if adj:
                    result[direction] = {
                        "page_num": adj["page_num"], "breadcrumb": adj["breadcrumb"],
                        "page_type": adj["page_type"], "content": adj["content"]
                    }

        return result


@mcp.tool()
def get_pages(doc_id: int, page_start: int, page_end: int) -> ToolResult:
    """Get full content of a range of pages. Max 10 pages per call.

    Args:
        doc_id: Document ID.
        page_start: First page number (inclusive).
        page_end: Last page number (inclusive).
    """
    with get_db() as conn:
        d = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not d:
            return {"error": f"Document {doc_id} not found"}

        page_end = min(page_end, page_start + 9)  # cap at 10 pages

        rows = conn.execute(
            "SELECT * FROM pages WHERE doc_id = ? AND page_num BETWEEN ? AND ? "
            "ORDER BY page_num",
            (doc_id, page_start, page_end)
        ).fetchall()

        if not rows:
            return {
                "error": f"No pages found in range {page_start}-{page_end}",
                "total_pages": d["total_pages"]
            }

        return {
            "doc_id": doc_id, "doc_title": d["title"], "project": d["project"],
            "total_pages": d["total_pages"],
            "pages": [
                {"page_num": p["page_num"], "breadcrumb": p["breadcrumb"],
                 "page_type": p["page_type"], "content": p["content"]}
                for p in rows
            ]
        }


@mcp.tool()
def get_adjacent(doc_id: int, page_num: int, direction: str = "next") -> ToolResult:
    """Get the next or previous page.

    Args:
        doc_id: Document ID.
        page_num: Current page number.
        direction: "next" or "prev".
    """
    target = page_num + (1 if direction == "next" else -1)
    return get_page(doc_id, target)


# ===== Fallback extraction tools =====

@mcp.tool()
def reextract_page(doc_id: int, page_start: int, page_end: Optional[int] = None) -> ToolResult:
    """Re-extract text from original PDF using pdfplumber (layout-preserved).

    Use this when Marker's output for a page looks garbled, has missing text,
    or seems to have OCR errors. This goes back to the source PDF for a fresh extraction.

    Args:
        doc_id: Document ID.
        page_start: First page to re-extract (1-indexed).
        page_end: Last page (default: same as page_start).
    """
    if page_end is None:
        page_end = page_start

    with get_db() as conn:
        d = conn.execute("SELECT pdf_path, title FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not d:
            return {"error": f"Document {doc_id} not found"}
        if not d["pdf_path"] or not Path(d["pdf_path"]).exists():
            return {"error": f"Original PDF not found at {d['pdf_path']}"}

    try:
        import pdfplumber
    except ImportError:
        return {"error": "pdfplumber not installed. Run: pip install pdfplumber"}

    try:
        results = []
        with pdfplumber.open(d["pdf_path"]) as pdf:
            for pg_num in range(page_start, min(page_end + 1, len(pdf.pages) + 1)):
                page = pdf.pages[pg_num - 1]
                text = page.extract_text(layout=True, x_tolerance=3, y_tolerance=3) or ""
                results.append({
                    "page_num": pg_num,
                    "content": text,
                    "char_count": len(text),
                    "width": page.width,
                    "height": page.height
                })

        return {
            "doc_id": doc_id, "doc_title": d["title"],
            "source": "pdfplumber (layout mode)",
            "pages": results
        }

    except Exception as e:
        return {"error": f"Extraction failed: {e}"}


@mcp.tool()
def reextract_table(doc_id: int, page_start: int, page_end: Optional[int] = None) -> ToolResult:
    """Re-extract tables from original PDF using pdfplumber with structure detection.

    Use this when a table in Marker's output has misaligned columns, merged cells,
    or garbled content. Returns structured table data with identified headers and rows.

    Args:
        doc_id: Document ID.
        page_start: First page containing the table (1-indexed).
        page_end: Last page of the table (for multi-page tables).
    """
    if page_end is None:
        page_end = page_start

    with get_db() as conn:
        d = conn.execute("SELECT pdf_path, title FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not d:
            return {"error": f"Document {doc_id} not found"}
        if not d["pdf_path"] or not Path(d["pdf_path"]).exists():
            return {"error": f"Original PDF not found at {d['pdf_path']}"}

    try:
        import pdfplumber
    except ImportError:
        return {"error": "pdfplumber not installed. Run: pip install pdfplumber"}

    try:
        all_tables = []

        with pdfplumber.open(d["pdf_path"]) as pdf:
            for pg_num in range(page_start, min(page_end + 1, len(pdf.pages) + 1)):
                page = pdf.pages[pg_num - 1]

                # Try strict line-based extraction first
                tables = page.extract_tables({
                    "vertical_strategy": "lines_strict",
                    "horizontal_strategy": "lines_strict",
                    "snap_tolerance": 5,
                    "join_tolerance": 5,
                })

                if not tables:
                    tables = page.extract_tables({
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 5,
                    })

                for raw_table in (tables or []):
                    if not raw_table or len(raw_table) < 2:
                        continue

                    # Clean cells
                    cleaned = []
                    for row in raw_table:
                        cleaned.append([
                            (str(c).strip().replace('\n', ' ') if c else '')
                            for c in row
                        ])

                    # Detect headers (first row usually)
                    headers = cleaned[0]
                    data = cleaned[1:]

                    # Build markdown
                    cols = max(len(r) for r in cleaned)
                    headers = headers + [''] * (cols - len(headers))
                    md_lines = [
                        '| ' + ' | '.join(headers) + ' |',
                        '| ' + ' | '.join(['---'] * cols) + ' |',
                    ]
                    for row in data:
                        padded = row + [''] * (cols - len(row))
                        md_lines.append('| ' + ' | '.join(padded) + ' |')

                    all_tables.append({
                        "page_num": pg_num,
                        "headers": headers,
                        "rows": data,
                        "row_count": len(data),
                        "col_count": cols,
                        "markdown": '\n'.join(md_lines)
                    })

        return {
            "doc_id": doc_id, "doc_title": d["title"],
            "source": "pdfplumber (table extraction)",
            "page_range": f"{page_start}-{page_end}",
            "tables_found": len(all_tables),
            "tables": all_tables
        }

    except Exception as e:
        return {"error": f"Table extraction failed: {e}"}


# ===== Semantic / vector search =====

_embedding_model = None
_embedding_model_name = None


def _get_embedding_model(model_name: str = 'all-MiniLM-L6-v2'):
    global _embedding_model, _embedding_model_name
    if _embedding_model is None or _embedding_model_name != model_name:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(model_name)
        _embedding_model_name = model_name
    return _embedding_model


@mcp.tool()
def semantic_search(
    project: str,
    query: str,
    doc_id: Optional[int] = None,
    max_results: int = 10
) -> ToolResult:
    """Semantic (vector) search — finds conceptually similar content even when
    exact keywords don't match. Use this when search_pages returns too few results
    or when you don't know the exact terminology used in the documents.

    Requires embeddings to have been computed during ingestion
    (pip install sentence-transformers).

    Args:
        project: Folder path. Includes sub-folders.
        query: Natural language query — can be a full question or description.
        doc_id: Optional. Restrict to one document.
        max_results: Default 10, max 25.
    """
    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy is required. Install: pip install numpy"}

    try:
        _get_embedding_model()
    except Exception:
        return {
            "error": "semantic search requires sentence-transformers. "
                     "Install: pip install sentence-transformers"
        }

    cap = min(max_results, 25)

    with get_db() as conn:
        ids = project_doc_ids(conn, project)
        if not ids:
            return {"error": f"No documents in folder '{project}'"}

        # Check if embeddings exist
        ph = ','.join('?' * len(ids))
        count = conn.execute(f"""
            SELECT COUNT(*) FROM page_embeddings pe
            JOIN pages p ON p.id = pe.page_id
            WHERE p.doc_id IN ({ph})
        """, ids).fetchone()[0]

        if count == 0:
            return {
                "error": "No embeddings found. Run: python ingest.py --embed-only --project <dir> --db <db>"
            }

        model_row = conn.execute(f"""
            SELECT pe.model FROM page_embeddings pe
            JOIN pages p ON p.id = pe.page_id
            WHERE p.doc_id IN ({ph}) LIMIT 1
        """, ids).fetchone()
        model = _get_embedding_model(model_row['model'])
        query_emb = model.encode([query])[0]

        doc_filter = ""
        extra_params = list(ids)
        if doc_id is not None:
            doc_filter = "AND p.doc_id = ?"
            extra_params.append(doc_id)

        rows = conn.execute(f"""
            SELECT pe.page_id, pe.chunk_index, pe.chunk_text, pe.embedding,
                   p.doc_id, p.page_num, p.breadcrumb, p.page_type,
                   d.title as doc_title, d.total_pages
            FROM page_embeddings pe
            JOIN pages p ON p.id = pe.page_id
            JOIN documents d ON d.id = p.doc_id
            WHERE p.doc_id IN ({ph}) {doc_filter}
        """, extra_params).fetchall()

        dim = len(query_emb)
        scored = []
        for row in rows:
            emb = np.frombuffer(row['embedding'], dtype=np.float32)
            if len(emb) != dim:
                continue
            sim = float(np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8
            ))
            scored.append({
                'doc_id': row['doc_id'],
                'doc_title': row['doc_title'],
                'page_num': row['page_num'],
                'total_pages': row['total_pages'],
                'breadcrumb': row['breadcrumb'],
                'page_type': row['page_type'],
                'chunk_preview': row['chunk_text'][:200],
                'similarity': round(sim, 4),
            })

        scored.sort(key=lambda x: x['similarity'], reverse=True)

        # Deduplicate by (doc_id, page_num) — keep highest similarity
        seen = set()
        deduped = []
        for r in scored:
            key = (r['doc_id'], r['page_num'])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
                if len(deduped) >= cap:
                    break

        return {
            "query": query,
            "result_count": len(deduped),
            "results": deduped,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global DB_PATH, SSE_PORT

    p = argparse.ArgumentParser(description='Document RAG MCP Server')
    p.add_argument('--db', '-d', required=True, help='SQLite database path')
    p.add_argument('--port', type=int, default=8200, help='SSE port (default: 8200)')

    args = p.parse_args()
    DB_PATH = args.db

    if not Path(DB_PATH).exists():
        print(f"Error: {DB_PATH} not found. Run ingest.py first.")
        sys.exit(1)

    # Update port on the mcp instance
    mcp.settings.port = args.port

    with get_db() as conn:
        folders = conn.execute(
            "SELECT project, COUNT(*) as n FROM documents GROUP BY project"
        ).fetchall()
        total_pages = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]

        print(f"Document RAG MCP Server")
        print(f"Database: {DB_PATH}")
        folder_list = ', '.join(f"{r['project']} ({r['n']} docs)" for r in folders)
        print(f"Folders: {folder_list}")
        print(f"Total pages indexed: {total_pages}")
        print(f"Starting on port {args.port}...")

    mcp.settings.host = "0.0.0.0"
    mcp.settings.transport_security.enable_dns_rebinding_protection = False
    mcp.run(transport="sse")


if __name__ == '__main__':
    main()
