"""OCR text and expected fields for three real test invoices (test data only).

The ``*_PSM6`` and ``*_PSM4`` strings are the exact raw outputs of
``extract_text(preprocess_image(image), psm=...)`` for the sample invoices, so
they include OCR mistakes such as the corrupted label ``ice No``.
"""

NORTHSTAR_PSM6 = 'Northstar Office Supplies INVOICE\n\nPlot 21, Sector I-9 Industrial Area | ice No: INV-2026-00125\n\nIslamabad 44790, Pakistan noice 0: : .\n\nTel: +92 51 555 0142 | NTN: 1234567-8 Invoice Date: 24 October 2026\nTerms: Net 30 days\n\nBILL TO\n\nZenith Educational Trust\n\nHouse 7, Street 15, F-8/3\n\nIslamabad, Pakistan\n\ncr)\n\nA4 Copier Paper {ream} 40 650.00 26,000.00\n\nBallpoint Pens {box of 50) 480.00 12,000.00\n\nHeavy Duty Stapler 1,250.00 12,500.00\n\nfa Manila File Folders {pack} 120.00 6,000.00\n\nPayment Terms: Payment due within 30 days of the invoice date. 3\n\nPlease quote the invoice number with your payment. GST (18%) 10,170.00\n\nBank: Example Commercial Bank | A/C No: 0011 2233 4455 66,670.00\n\nThank you for your business.\n'

NORTHSTAR_PSM4 = 'Northstar Office Supplies INVOICE\n\nPlot 21, Sector I-9 Industrial Area . .\nIslamabad 44790, Pakistan Invoice No: INV-2026-00125\n\nTel: +92 51 555 0142 | NTN: 1234567-8 Invoice Date: 24 October 2026\nTerms: Net 30 days\n\nBILL TO\n\nZenith Educational Trust\nHouse 7, Street 15, F-8/3\nIslamabad, Pakistan\n\nNo. | Description Qty Unit Price (PKR) Amount (PKR)\n1 | A4 Copier Paper (ream) 40 650.00 26,000.00\n2 | Ballpoint Pens {box of 50) 25 480.00 12,000.00\n3 | Heavy Duty Stapler 10 1,250.00 12,500.00\n4 | Manila File Folders {pack} 50 120.00 6,000.00\n| Subtotal | 56,500.00\n\nPayment Terms: Payment due within 30 days of the invoice date. 3\nPlease quote the invoice number with your payment. | GST (18%) | 10,170.00\nBank: Example Commercial Bank | A/C No: 0011 2233 4455 | Total | 66,670.00\n\nThank you for your business.\n'

BLUEPEAK_PSM6 = 'BluePeak Technologies INVOICE\n\nCloud & Software Services\n\nINVOICE NUMBER ISSUE DATE DUE DATE\n\nBP/2026/0842 2026-09-15 2026-10-15\n\nBILLED TO FROM\n\nRedwood Analytics Inc. 200 Harbor Blvd, Suite 410\n\n88 Congress Ave, Floor 6 Austin, TX 78701, USA\n\nAustin, TX 78701 billing@bluepeak-tech.example\n\nDESCRIPTION QTY UNIT PRICE AMOUNT\n\nCloud Hosting (months) 3 $450.00 $1,350.00\n\nSSL Certificate (annual) 1 $130.00 $130.00\n\nSupport Retainer (hours) 10 $95.00 $950.00\nSubtotal $2,430.00\nSales Tax (8.5%) $206.55\nTotal Due $2,636.55\n\nAll amounts in USD\nPlease pay by bank transfer to the account on file.\nThank you for choosing BluePeak.\n'

BLUEPEAK_PSM4 = 'BluePeak Technologies INVOICE\n\nCloud & Software Services\n\nINVOICE NUMBER ISSUE DATE DUE DATE\n\nBP/2026/0842 2026-09-15 2026-10-15\n\nBILLED TO FROM\n\nRedwood Analytics Inc. 200 Harbor Blvd, Suite 410\n\n88 Congress Ave, Floor 6 Austin, TX 78701, USA\n\nAustin, TX 78701 billing@bluepeak-tech.example\n\nDESCRIPTION QTY UNIT PRICE AMOUNT\n\nCloud Hosting (months) 3 $450.00 $1,350.00\n\nSSL Certificate (annual) 1 $130.00 $130.00\n\nSupport Retainer (hours) 10 $95.00 $950.00\nSubtotal $2,430.00\nSales Tax (8.5%) $206.55\nTotal Due $2,636.55\n\nAll amounts in USD\n\nPlease pay by bank transfer to the account on file.\nThank you for choosing BluePeak.\n'

GREENFIELD_PSM6 = 'Greenfield Trading\nWarehouse 9, Gulberg Industrial Estate\nLahore 54000, Pakistan Net 30\nTel: +92 42 555 0198 | NTN 7654321-0 Customer PO PO-55012\nBill To: Ship To:\nAl-Madina Superstores Al-Madina Superstores\nMain Boulevard, Model Town Main Boulevard, Model Town\nLahore, Pakistan Lahore, Pakistan\n[Code —_| Description Unit Price (PKR) Amount (PKR)\nGT-101 | Basmati Rice 25kg Bag 4,200.00 42,000.00\nGT-102 | Refined Sugar 50kg Bag 8,600.00 43,000.00\nGT-103 | Cooking Oil SL Can 2,150.00 51,600.00\nGT-104 | Black Tea 1kg Pack 1,850.00 22,200.00\nGT-105 | Red Lentils 25kg Bag | 6 3,900.00 23,400.00\nGT-106 _ | lodized Salt 1kg Pack | 40 85.00 3,400.00\nGT-107 | Wheat Flour 5¢kg Bag | al 5,300.00 42,400.00\nItems: 7 | Total Qty: 105 | Delivery: Same day\nNotes: Goods checked and received in good condition. Sub-Total PKR 228,000.00\nBank: Example Commercial Bank, Lahore | Acct 0123456789 Tax (5%) PKR 11,400.00\nCheques payable to: Greenfield Trading PKR 239,400.00\nTerms and Conditions: Payment is due within 30 days. Overdue balances may attract a surcharge of 2% per month.\nGoods once sold are not returnable except for manufacturing defects reported within 7 days of delivery.\nTitle to the goods remains with the seller until payment is received in full. Disputes are subject to Lahore courts.\nThis is a computer generated invoice and does not require a signature.\n'

GREENFIELD_PSM4 = '| Invoice No. | GS-26-000731\n\nGreenfield Trading\n\n; | Date | 12-Oct-2026\nWarehouse 9, Gulberg Industrial Estate\nLahore 54000, Pakistan | Terms | Net 30\nTel: +92 42 555 0198 | NTN 76543210 | Customer PO | PO-55012\nBill To: Ship To:\nAl-Madina Superstores Al-Madina Superstores\nMain Boulevard, Model Town Main Boulevard, Model Town\nLahore, Pakistan Lahore, Pakistan\n\nDescription Unit Price (PKR) Amount (PKR)\n\nGT-101 Basmati Rice 25kg Bag 4,200.00 42,000.00\nGT-102 Refined Sugar 50kg Bag 5 8,600.00 43,000.00\nGT-103 Cooking Oil SL Can 24 2,150.00 51,600.00\nGT-104 Black Tea 1kg Pack 12 1,850.00 22,200.00\nGT-105 Red Lentils 25kg Bag 6 3,900.00 23,400.00\nGT-106 lodized Salt 1kg Pack 40 85.00 3,400.00\nGT-107 Wheat Flour 50kg Bag 8 5,300.00 42,400.00\nItems: 7 | Total Qty: 105 | Delivery: Same day\n\nNotes: Goods checked and received in good condition. Sub-Total PKR 228,000.00\nBank: Example Commercial Bank, Lahore | Acct 0123456789 | Tax (5%) PKR 11,400.00\nCheques payable to: Greenfield Trading | Total PKR 239,400.00\n\nTerms and Conditions: Payment is due within 30 days. Overdue balances may attract a surcharge of 2% per month.\nGoods once sold are not returnable except for manufacturing defects reported within 7 days of delivery.\n\nTitle to the goods remains with the seller until payment is received in full. Disputes are subject to Lahore courts.\nThis is a computer generated invoice and does not require a signature.\n'

NORTHSTAR_EXPECTED = {
    "invoice_number": "INV-2026-00125",
    "invoice_date": "24 October 2026",
    "vendor_name": "Northstar Office Supplies",
    "subtotal": "56500.00",
    "tax_amount": "10170.00",
    "total_amount": "66670.00",
}

BLUEPEAK_EXPECTED = {
    "invoice_number": "BP/2026/0842",
    "invoice_date": "2026-09-15",
    "vendor_name": "BluePeak Technologies",
    "subtotal": "2430.00",
    "tax_amount": "206.55",
    "total_amount": "2636.55",
}

GREENFIELD_EXPECTED = {
    "invoice_number": "GS-26-000731",
    "invoice_date": "12-Oct-2026",
    "vendor_name": "Greenfield Trading",
    "subtotal": "228000.00",
    "tax_amount": "11400.00",
    "total_amount": "239400.00",
}
