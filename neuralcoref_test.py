#encoding:utf-8

import spacy
spacy_nlp=spacy.load('en_core_web_md')

import neuralcoref
neuralcoref.add_to_pipe(spacy_nlp)

doc="XXE vulnerability exists in the Metasys family of product Web Services " \
    "which has the potential to facilitate DoS attacks or harvesting of ASCII server files. " \
    "This affects Johnson Controls' Metasys Application and Data Server (ADS, ADS-Lite) versions 10.1 and prior; " \
    "Metasys Extended Application and Data Server (ADX) versions 10.1 and prior; Metasys Open Data Server (ODS) versions 10.1 " \
    "and prior; Metasys Open Application Server (OAS) version 10.1; Metasys Network Automation Engine (NAE55 only) versions 9.0.1, " \
    "9.0.2, 9.0.3, 9.0.5, 9.0.6; Metasys Network Integration Engine (NIE55/NIE59) versions 9.0.1, 9.0.2, 9.0.3, 9.0.5, 9.0.6; " \
    "Metasys NAE85 and NIE85 versions 10.1 and prior; Metasys LonWorks Control Server (LCS) versions 10.1 and prior; " \
    "Metasys System Configuration Tool (SCT) versions 13.2 and prior; Metasys Smoke Control Network Automation Engine " \
    "(NAE55, UL 864 UUKL/ORD-C100-13 UUKLC 10th Edition Listed) version 8.1."

doc1='"IBM DB2 9.7, 10,1, 10.5, and 11.1 is vulnerable to an unauthorized command ' \
     'that allows the database to be activated when authentication type is CLIENT.'
'''
期望输出：
    软件名称：IBM DB2
    软件版本：9.7, 10,1, 10.5, and 11.1
    漏洞位置：
    利用条件：
    漏洞类型：
    修复情况：无
    产生影响：
'''

doc2='net/packet/af_packet.c in the Linux kernel before 4.13.6 ' \
     'allows local users to gain privileges via crafted system calls ' \
     'that trigger mishandling of packet_fanout data structures, ' \
     'because of a race condition (involving fanout_add and packet_do_bind) ' \
     'that leads to a use-after-free, a different vulnerability than CVE-2017-6346.'
'''
期望输出：
    软件名称：Linux kernel
    软件版本：4.13.6
    漏洞位置：net/packet/af_packet.c
    利用条件：crafted system calls
    漏洞类型：allow local users to gain privileges
    修复情况：无
    产生影响：use-after-free
'''

print(doc._.coref_resolved)