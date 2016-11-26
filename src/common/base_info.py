p1_clsval2label ={
    0:'SNE',
    1:'LY',
    2:'MO',
    3:'BNE',
    4:'EO',
    5:'BA',
    6:'MY',
    7:'MMY',
    8:'ERB',
    9:'BL',
    10:'PMY',
    11:'GT',
    12:'ART',
    13:'SMU',
    14:'TAG',
    15:'VLY',
    16:'PC',
    17:'OTH',
    18:'MEK',
}

p1_label2clsval ={
    'SNE':0,
    'LY':1,
    'MO':2,
    'BNE':3,
    'EO':4,
    'BA':5,
    'MY':6,
    'MMY':7,
    'ERB':8,
    'BL':9,
    'PMY':10,
    'GT':11,
    'ART':12,
    'SMU':13,
    # 'TAG':14,
    'VLY':15,
    'PC':16,
    'NC':17,
    'OTH':17,
    # 'UI':17,
    'MEK':18,
    'ERC':14,  # TAGがERCと間違えてラベル付けされていたため、ERCも14としている
}

p1_label2trial_label ={
    'SNE':'A',
    'LY':'B',
    'MO':'C',
    'BNE':'D',
    'EO':'E',
    'BA':'F',
    'MY':'G',
    'MMY':'H',
    'ERB':'I',
    'BL':'J',
    'PMY':'K',
    'GT':'L',
    'ART':'M',
    'SMU':'N',
    # 'TAG':14,
    'VLY':'P',
    'PC':'Q',
    'NC':'R',
    'OTH':'R',
    # 'UI':17,
    'MEK':'S',
    'ERC':'O',  # TAGがERCと間違えてラベル付けされていたため、ERCも14としている
}


def get_base_info(fetch_type='all'):
    if fetch_type=='all':
        return p1_clsval2label, p1_label2clsval, p1_label2trial_label
    elif fetch_type=='clsval2labstr':
        return p1_clsval2label
    elif fetch_type=='labstr2clsval':
        return p1_label2clsval
    elif fetch_type=='label2trial_label':
        return p1_label2trial_label
