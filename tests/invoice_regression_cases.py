"""Ten-invoice regression data (test data only).

Each case holds the raw OCR text of one real test invoice at PSM 6 (the primary
pass) and PSM 4 (the fallback pass), plus the fields the application must
return. Invoices 1-3 come from ``invoice_ocr_samples``. Invoices 4-7 are the raw
``extract_text(preprocess_image(image), psm=...)`` output of the invoice images
(Tesseract 5.3.4), so they keep OCR mistakes such as ``PKR 1.250,000.00``.

To add an invoice: paste its two raw OCR texts, write down the expected fields
as the invoice shows them, and append a ``Case`` to ``CASES``. Dates and invoice
numbers are expected exactly as written on the invoice; amounts are decimal
strings. A field no OCR pass can read reliably is expected to be ``None``.
"""

from dataclasses import dataclass

from invoice_ocr_samples import (
    BLUEPEAK_EXPECTED,
    BLUEPEAK_PSM4,
    BLUEPEAK_PSM6,
    GREENFIELD_EXPECTED,
    GREENFIELD_PSM4,
    GREENFIELD_PSM6,
    NORTHSTAR_EXPECTED,
    NORTHSTAR_PSM4,
    NORTHSTAR_PSM6,
)


@dataclass(frozen=True)
class Case:
    name: str
    psm6: str
    psm4: str
    expected: dict[str, str | None]
    # The first pass gives all six fields without needing the fallback.
    needs_fallback: bool
    # PSM 4 on the 0.75x copy of the page (the third pass). Empty if never needed.
    psm4_rescaled: str = ""
    # The invoice still has an empty field after PSM 6 and PSM 4, so the third pass runs.
    needs_rescale: bool = False


HARBORVIEW_PSM6 = 'Harborview Wholesale\n\n14 Dockside Road\n\nPortland, OR 97209, USA\n\norders@harborview-wholesale.example\n\nInvoice No. HV-2026-4408\n\nInvoice Date Oct 03, 2026\n\nDue Date Nov 02, 2026\n\nPayment Terms Net 30\n\nBill To\n\nPine & Bean Cafe\n\n402 NW 21st Avenue\n\nPortland, OR 97209\nItem Description Quantity Rate (USD) Amount (USD)\nRoasted Coffee Beans 5lb Bag 20 42.00 840.00\nPaper Cups 120z (case) 15 36.50 547.50\nPaper Napkins (case) 30 12.25 367.50\nNet Amount Tax Amount Due\nUSD 1755.00 USD 105.30 USD 1860.30\n\nTax rate applied: 6%\nRemit payment to: Harborview Wholesale, Account 7788 9900, Example Bank, Portland OR.\nLate payments are subject to a 1.5% monthly service charge.\n'

HARBORVIEW_PSM4 = 'INVOICE\n\nHarborview Wholesale\n\n14 Dockside Road\nPortland, OR 97209, USA\norders@harborview-wholesale.example\n\nInvoice No. HV-2026-4408\nInvoice Date Oct 03, 2026\nDue Date Nov 02, 2026\nPayment Terms Net 30\n\nBill To\n\nPine & Bean Cafe\n402 NW 21st Avenue\nPortland, OR 97209\n\nItem Description Quantity Rate (USD) Amount (USD)\nRoasted Coffee Beans 5lb Bag 20 42.00 840.00\nPaper Cups 120z (case) 15 36.50 547.50\nPaper Napkins (case) 30 12.25 367.50\nNet Amount Tax Amount Due\n\nUSD 1755.00 USD 105.30 USD 1860.30\n\nTax rate applied: 6%\n\nRemit payment to: Harborview Wholesale, Account 7788 9900, Example Bank, Portland OR.\nLate payments are subject to a 1.5% monthly service charge.\n'

ALPINE_PSM6 = 'Alpine Digital Services GmbH INVOICE\nLeopoldstrasse 88 Invoice No.: RE-2026-10-0173\n80802 Munich, Germany Invoice Date: 19.10.2026\nUStIdNr.: DE123456789 nvolce ate. S920.\nDue Date: 02.11.2026\nCurrency: EUR\nCustomer\nBrandt Maschinenbau AG\nHafenstrasse 12\n20457 Hamburg, Germany\nPos. Description Qty Unit Price (EUR) Amount (EUR)\nWeb Application Development (hours) 85,00 3.400,00\nServer Maintenance (months) 45,00 540,00\nUX Audit (fixed fee) 1.250,00 1.250,00\nNumber format on this invoice: Net Amount EUR 5.190,00\nperiod (.) = thousands separator, 5\ncomma (,) = decimal separator. VAT 19% EUR 986,10\nExample: 1.234,56 = 1234 euros and 56 cents. Total Amount EUR 6.176.10\nPayment: IBAN DEOO 0000 0000 0000 0000 00 | BIC EXMPDEMMXXX\n'

ALPINE_PSM4 = 'Alpine Digital Services GmbH INVOICE\n\nLeopoldstrasse 88\n80802 Munich, Germany\nUSt-IdNr.: DE123456789\n\nInvoice No.: RE-2026-10-0173\nInvoice Date: 19.10.2026\n\nDue Date: 02.11.2026\nCurrency: EUR\n\nCustomer\n\nBrandt Maschinenbau AG\nHafenstrasse 12\n\n20457 Hamburg, Germany\n\nPos. Description Qty Unit Price (EUR) Amount (EUR)\n1 Web Application Development (hours) 40 85,00 3.400,00\n\n2 Server Maintenance (months) 12 45,00 540,00\n\n3 UX Audit (fixed fee) 1 1.250,00 1.250,00\nNumber format on this invoice: Net Amount EUR 5.190,00\n\nperiod (.) = thousands separator,\ncomma (,) = decimal separator.\nExample: 1.234,56 = 1234 euros and 56 cents.\n\nVAT 19% EUR 986,10\n\nTotal Amount EUR 6.176,10\n\nPayment: IBAN DEOO 0000 0000 0000 0000 00 | BIC EXMPDEMMXXX\n'

VERTEX_PSM6 = "Vertex Consulting TAX INVOICE\nLevel 4, 22 King's Row Invoice No: VX/GBR/2026/0291\nManchester M2 4WU, United Kingdom Invoice Date: 14/10/2026\nVAT Reg No: GB 123 4567 89 Due Date: 13/11/2026\nBILLED TO\nCalder & Finch Retail Ltd\n9 Bridge Street\nLeeds LS1 4DY, United Kingdom\nee\nuosseenmseins | 8)\nVAT Summary Subtotal GBP 7,950.00\nfrm | netamoit| Arana\nVAT (20%) GBP 1,590.00\n\n7,950.00 1,590.00\n\noer 950\nSubtotal is the sum of all net line amounts. VAT is charged at 20% of the net amount.\nPayment terms: 30 days from invoice date.\n"

VERTEX_PSM4 = "Vertex Consulting TAX INVOICE\n\nLevel 4, 22 King's Row Invoice No: VX/GBR/2026/0291\nManchester M2 4WU, United Kingdom Invoice Date: 14/10/2026\n\nVAT Reg No: GB 123 4567 89 Due Date: 13/11/2026\n\nBILLED TO\n\nCalder & Finch Retail Ltd\n9 Bridge Street\nLeeds LS1 4DY, United Kingdom\n\nDescription Qty Unit Price Net (GBP) | VAT % VAT (GBP)\nStrategy Workshop (days) 2 1,200.00 2,400.00 20% 480.00\nMarket Analysis Report 1 1,850.50 1,850.50 20% 370.10\nStakeholder Interviews 6 175.00 1,050.00 20% 210.00\nProcess Mapping Sessions 3 420.00 1,260.00 20% 252.00\nTraining Session (hours) 4 250.00 1,000.00 20% 200.00\nTravel Expenses 1 389.50 389.50 20% 77.90\n\nVAT Summary | Subtotal GBP 7,950.00\nVAT Rate | Net Amount | VAT Amount |\n\n| VAT (20%) GBP 1,590.00\n20% | 7,950.00 | 1,590.00 |\n| Total GBP 9,540.00\n\nSubtotal is the sum of all net line amounts. VAT is charged at 20% of the net amount.\nPayment terms: 30 days from invoice date.\n"

CRESCENT_PSM6 = 'Crescent Industrial Solutions TAX INVOICE\nB-44, SITE Industrial Area, Karachi 75700, Pakistan\nNTN: 2468135-7 | Tel: +92 21 555 0176\n\nInvoice No: CS-INV-2026-019 | Date: 24/10/2026 | Due: 23/11/2026\nBilled to: Delta Steel Mills (Pvt) Ltd, Plot 14, Port Qasim Industrial Zone, Karachi\nOrder Ref: SO-77312 | Delivery Terms: Ex-Works Karachi | Payment: Bank Transfer\nDescription Qty Unit Price (PKR) Amount (PKR}\nHydraulic Pump Unit HP-450 5 125,000.00 625,000.00\nPressure Valve Set PV-12 20 18,750.00 375,000.00\nInstallation & Commissioning 1 250,000.00 250,000.00\nRemarks: Goods once sold will not be taken back. Subtotal PKR 1.250,000.00\nBank: Example Bank, Karachi | Acct: 0011223344\n\nTax Amount (18%) PKR 225,000.00\n'

CRESCENT_PSM4 = 'Crescent Industrial Solutions TAX INVOICE\n\nB-44, SITE Industrial Area, Karachi 75700, Pakistan\nNTN: 2468135-7 | Tel: +92 21 555 0176\n\nInvoice No: CS-INV-2026-019 | Date: 24/10/2026 | Due: 23/11/2026\n\nBilled to: Delta Steel Mills (Pvt) Ltd, Plot 14, Port Qasim Industrial Zone, Karachi\nOrder Ref: SO-77312 | Delivery Terms: Ex-Works Karachi | Payment: Bank Transfer\n\nDescription Qty Unit Price (PKR) Amount (PKR}\nHydraulic Pump Unit HP-450 5 125,000.00 625,000.00\nPressure Valve Set PV-12 20 18,750.00 375,000.00\nInstallation & Commissioning 1 250,000.00 250,000.00\nRemarks: Goods once sold will not be taken back. | Subtotal | PKR 1.250,000.00 |\n\nBank: Example Bank, Karachi | Acct: 0011223344\n\n| Tax Amount (18%) | PKR 225,000.00 |\n\n| Total | PKR 1,475,000.00 |\n\n'

# Crescent's third pass: the preprocessed page scaled to 0.75x, read at PSM 4 (Tesseract 5.3.4).
# Only at this scale are the commas of "PKR 1,250,000.00" read correctly.
CRESCENT_PSM4_RESCALED = 'Crescent Industrial Solutions TAX INVOICE\n\nB-44, SITE Industrial Area, Karachi 75700, Pakistan\nNTN: 2468135-7 | Tel: +92 21 555 0176\n\nInvoice No: CS-INV-2026-019 | Date: 24/10/2026 | Due: 23/11/2026\n\nBilled to: Delta Steel Mills (Pvt) Ltd, Plot 14, Port Qasim Industrial Zone, Karachi\nOrder Ref: SO-77312 | Delivery Terms: Ex-Works Karachi | Payment: Bank Transfer\n\nDescription Qty Unit Price (PKR) Amount (PKR)\nHydraulic Pump Unit HP-450 5 125,000.00 625,000.00\nPressure Valve Set PV-12 20 18,750.00 375,000.00\nInstallation & Commissioning 1 250,000.00 250,000.00\nRemarks: Goods once sold will not be taken back. Subtotal PKR 1,250,000.00\n\nBank: Example Bank, Karachi | Acct: 0011223344\nTax Amount (18%) PKR 225,000.00\n\nTotal PKR 1,475,000.00\n\n'

HARBORVIEW_EXPECTED = {
    "invoice_number": "HV-2026-4408",
    "invoice_date": "Oct 03, 2026",
    "vendor_name": "Harborview Wholesale",
    "subtotal": "1755.00",
    "tax_amount": "105.30",
    "total_amount": "1860.30",
}

ALPINE_EXPECTED = {
    "invoice_number": "RE-2026-10-0173",
    "invoice_date": "19.10.2026",
    "vendor_name": "Alpine Digital Services GmbH",
    "subtotal": "5190.00",
    "tax_amount": "986.10",
    "total_amount": "6176.10",
}

VERTEX_EXPECTED = {
    "invoice_number": "VX/GBR/2026/0291",
    "invoice_date": "14/10/2026",
    "vendor_name": "Vertex Consulting",
    "subtotal": "7950.00",
    "tax_amount": "1590.00",
    "total_amount": "9540.00",
}

# The image prints "PKR 1,250,000.00". At full size every OCR variant tried (PSM 3, 4, 6,
# 11 and 12; the preprocessed, grayscale and 2x images) reads the commas as periods
# ("1.250,000.00" or "1.250.000.00"), which is not a safe number, so PSM 6 and PSM 4
# leave it empty. The 0.75x third pass reads it correctly.
CRESCENT_EXPECTED = {
    "invoice_number": "CS-INV-2026-019",
    "invoice_date": "24/10/2026",
    "vendor_name": "Crescent Industrial Solutions",
    "subtotal": "1250000.00",
    "tax_amount": "225000.00",
    "total_amount": "1475000.00",
}

MAPLELINE_PSM6 = 'M1) Mapleline Furniture INVOICE\n77 Birchwood Lane\nBurlington, VT 05401, USA\nsales@mapleline-furniture.example\nBILL TO INVOICE DETAILS\nHartwell Hospitality Group Invoice No.: MF-0000819\n310 Lake Street Invoice Date: 30 Aug 2026\nBurlington, VT 05401 Order Ref: SO-77312\nDue Date: 29 Sep 2026\n# Item Qty Unit Price (USD) Amount (USD)\n1 Solid Oak Dining Table 4 $899.00 $3,596.00\n2 Upholstered Dining Chair 6 $349.50 $2,097.00\n3 Walnut Bookshelf 5-Tier 3 $215.00 $645.00\n4  White-Glove Delivery 1 $120.00 $120.00\n. ; . . Subtotal $6,458.00\nDelivery: White-glove delivery included above.\nThank you for your order. Sales Tax (7%) $452.06\nTOTAL PAYABLE $6,910.06\n'

MAPLELINE_PSM4 = 'Mapleline Furniture INVOICE\n\n77 Birchwood Lane\nBurlington, VT 05401, USA\nsales@mapleline-furniture.example\n\nBILL TO INVOICE DETAILS\n\nHartwell Hospitality Group Invoice No.: MF-0000819\n310 Lake Street Invoice Date: 30 Aug 2026\nBurlington, VT 05401 Order Ref: SO-77312\n\nDue Date: 29 Sep 2026\n\n# Item Qty Unit Price (USD) Amount (USD)\n1 Solid Oak Dining Table 4 $899.00 $3,596.00\n2 Upholstered Dining Chair 6 $349.50 $2,097.00\n3. Walnut Bookshelf 5-Tier 3 $215.00 $645.00\n4  White-Glove Delivery 1 $120.00 $120.00\n\nSubtotal $6,458.00\n\nDelivery: White-glove delivery included above.\nThank you for your order. Sales Tax (7%) $452.06\n\nTOTAL PAYABLE $6,910.06\n\n'

MAPLELINE_EXPECTED = {
    'invoice_number': 'MF-0000819',
    'invoice_date': '30 Aug 2026',
    'vendor_name': 'Mapleline Furniture',
    'subtotal': '6458.00',
    'tax_amount': '452.06',
    'total_amount': '6910.06'
}

MERIDIAN_PSM6 = 'Meridian Business Products INVOICE\nSuite 305, Blue Plaza, Jinnah Avenue\nSea eos 0173] NTN [inveice No. [20263NV-00058 |\nTel: +92 51 555 0123 | NTN: 9876543-2 Invoice No. 2026-INV-00058\n[bate | Sentember 17 2026\nOctober 02, 2026\nFiems Nets\nSold To:\nCapital Legal Associates\nOffice 12, 2nd Floor, Centaurus Mall\nIslamabad, Pakistan\nemanate | Ame\nLaser Printer Toner Cartridge 8,750.00 105,000.00\nMagnetic Whiteboard 4x3ft | 4 6,200.00 24,800.00\nDesk Organizer Set | 30! 1,450.00 43,500.00\nDate: GST (18%) 31,194.00\nGrand Total 204,494.00\nGoods received in good condition. Computer generated invoice.\n'

MERIDIAN_PSM4 = 'Meridian Business Products INVOICE\n\nSuite 305, Blue Plaza, Jinnah Avenue\n\nIslamabad 44000, Pakistan\n\nTel: +92 51 555 0123 | NTN: 9876543-2 : 2026-INV-00058\nSeptember 17, 2026\n\nOctober 02, 2026\nNet 15\n\nSold To:\n\nCapital Legal Associates\n\nOffice 12, 2nd Floor, Centaurus Mall\nIslamabad, Pakistan\n\nUnit Price (PKR) Amount (PKR)\n\nLaser Printer Toner Cartridge\n\nMagnetic Whiteboard 4x3ft\n\nDesk Organizer Set\n\n173,300.00\n\nSubtotal\n\nReceived by:\n\nGST (18%) 31,194.00\n204,494.00\n\nDate:\n\nGrand Total\n\nGoods received in good condition. Computer generated invoice.\n'

MERIDIAN_PSM4_RESCALED = 'Meridian Business Products INVOICE\n\nSuite 305, Blue Plaza, Jinnah Avenue\nIslamabad 44000, Pakistan Invoice No. 2026-INV-00058\n\nTel: +92 51 555 0123 | NTN: 9876543-2\nDate September 17, 2026\n\nDue Date October 02, 2026\n\nTerms Net 15\nSold To:\nCapital Legal Associates\nOffice 12, 2nd Floor, Centaurus Mall\nIslamabad, Pakistan\n\nQy Unit Price (PKR) Amount (PKR)\n\nDescription\n\nLaser Printer Toner Cartridge 12 8,750.00 105,000.00\nMagnetic Whiteboard 4x3ft | 4 6,200.00 24,800.00\nDesk Organizer Set 30 1,450.00 | 43,500.00\n\nReceived by: Subtotal 173,300.00\nDate: GST (18%) 31,194.00\nGrand Total | 204,494.00\n\nGoods received in good condition. Computer generated invoice.\n'

MERIDIAN_EXPECTED = {
    'invoice_number': '2026-INV-00058',
    'invoice_date': 'September 17, 2026',
    'vendor_name': 'Meridian Business Products',
    'subtotal': '173300.00',
    'tax_amount': '31194.00',
    'total_amount': '204494.00'
}

SILVERSTONE_PSM6 = 'Silverstone Electronics INVOICE\nUnit 6, Kingsway Trade Park\nBirmingham B1 1AA\nUnited Kingdom Bill To\nVAT Reg No: GB 987 6543 21\nNorthgate Learning Centre\n15 Priory Road\nCoventry CV1 2AB, United Kingdom\nINVOICE NO.\nSEL-2610-A0733\nINVOICE DATE Item Qty Price (GBP) Amount (GBP)\n05 OCT 2026\nWireless Router AX6000 5 189.99 949.95\nDUE DATE 24-Port Network Switch 3 245.00 735.00\n-Port Network Switcl . .\n19 OCT 2026\nCAT6 Cable Box {305 m)} 10 64.50 645.00\nCUSTOMER ID\nC-20931 Extended Warranty (1 year} 5 39.00 195.00\nPAYMENT TERMS\nNet 14 days Taxable Amount GBP 2,524.95\nVAT (20%) GBP 504.99\nBalance Due GBP 3,029.94\nPayment due within 14 days of the invoice date.\nThank you for your custom.\n'

SILVERSTONE_PSM4 = 'Silverstone Electronics INVOICE\n\nUnit 6, Kingsway Trade Park\nBirmingham B1 1AA\n\nUnited Kingdom\n\nVAT Reg No: GB 987 6543 21\n\nBill To\n\nNorthgate Learning Centre\n\n15 Priory Road\n\nCoventry CV1 2AB, United Kingdom\nINVOICE NO.\n\nSEL-2610-A0733\n\nINVOICE DATE Item Qty Price (GBP) Amount (GBP)\n05 OCT 2026\nWireless Router AX6000 5 189.99 949.95\nDUE DATE ;\n19 OCT 2026 24-Port Network Switch 3 245.00 735.00\nCAT6 Cable Box {305 m)} 10 64.50 645.00\nCUSTOMER ID\nC-20931 Extended Warranty (1 year} 5 39.00 195.00\nPAYMENT TERMS\nNet 14 days Taxable Amount GBP 2,524.95\nVAT (20%) GBP 504.99\nBalance Due GBP 3,029.94\n\nPayment due within 14 days of the invoice date.\nThank you for your custom.\n'

SILVERSTONE_EXPECTED = {
    'invoice_number': 'SEL-2610-A0733',
    'invoice_date': '05 OCT 2026',
    'vendor_name': 'Silverstone Electronics',
    'subtotal': '2524.95',
    'tax_amount': '504.99',
    'total_amount': '3029.94'
}

CASES = (
    Case("1_northstar", NORTHSTAR_PSM6, NORTHSTAR_PSM4, NORTHSTAR_EXPECTED, needs_fallback=True),
    Case("2_bluepeak", BLUEPEAK_PSM6, BLUEPEAK_PSM4, BLUEPEAK_EXPECTED, needs_fallback=False),
    Case("3_greenfield", GREENFIELD_PSM6, GREENFIELD_PSM4, GREENFIELD_EXPECTED, needs_fallback=True),
    Case("4_harborview", HARBORVIEW_PSM6, HARBORVIEW_PSM4, HARBORVIEW_EXPECTED, needs_fallback=False),
    Case("5_alpine", ALPINE_PSM6, ALPINE_PSM4, ALPINE_EXPECTED, needs_fallback=True),
    Case("6_vertex", VERTEX_PSM6, VERTEX_PSM4, VERTEX_EXPECTED, needs_fallback=True),
    Case("7_crescent", CRESCENT_PSM6, CRESCENT_PSM4, CRESCENT_EXPECTED, needs_fallback=True,
         psm4_rescaled=CRESCENT_PSM4_RESCALED, needs_rescale=True),
    Case("8_mapleline", MAPLELINE_PSM6, MAPLELINE_PSM4, MAPLELINE_EXPECTED, needs_fallback=False),
    Case("9_meridian", MERIDIAN_PSM6, MERIDIAN_PSM4, MERIDIAN_EXPECTED, needs_fallback=True, psm4_rescaled=MERIDIAN_PSM4_RESCALED, needs_rescale=True),
    Case("10_silverstone", SILVERSTONE_PSM6, SILVERSTONE_PSM4, SILVERSTONE_EXPECTED, needs_fallback=False),
)
