"""
Script to parse and process FIS ski jumping documents.

@author: wrotki8778
"""
import os
import os.path
import re
from datetime import datetime
import math
import glob
import pandas as pd
import numpy as np
from tika import parser
os.chdir('C:/Users/kubaf/Documents/Skoki')


def is_number(s):
    """Check that s can be described as a number or not."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def decimal(list_n, list_k):
    """
    Check that a list of numbers are multiples of corresponding values.

    Inputs:
        list - a list of numbers to check,
        list_k - a list of integers (1,2,5,10)
        which corresponds to the multiples of
        1, 0.5, 0.2 and 0.1 respectively

    Output:
        output - a list of True/False values
        and True is given iff an appropriate
        value of list is a multiple of an appropriate
        quantity in list_k.
    """
    output = []
    for i, number in enumerate(list_n):
        output = output + [abs(list_k[i]*number - int(list_k[i]*number)) > 0]
    return output


def take_number(string):
    """Return a first numeric value in a given string."""
    string = string.replace('m', '')
    string = string.replace('/', ' ')
    tmp = string.split(' ')
    take = [x for x in tmp if is_number(x)]
    return float(take[0])


def to_date(day, month, year):
    """Return a date format with given inputs.

    Inputs:
        day - number (01-31),
        month - string ('Jan', 'Feb' etc.)
        year - number (1000-9999),

    Output:
        a date-time expression
    """
    string = str(day)+' '+month+' '+str(year)
    return datetime.strptime(string, '%d %B %Y')


def validate(date_text):
    """Check that a date_text string can be described as in a date format."""
    if len(date_text) == 10:
        date_text = '0'+date_text
    if len(date_text) < 6:
        return False
    test = date_text[:3] + date_text[3].upper() + date_text[4:]
    try:
        if test != datetime.strptime(test, '%d %b %Y').strftime('%d %b %Y'):
            raise ValueError
        return True
    except ValueError:
        return False


def rozdziel(string):
    """Process a string to split a multi-'.'-substrings."""
    new_string = string
    if string.count('-'):
        new_string = string.split('-')[0]+' '+' '.join(['-'+e for e in string.split('-')[1:] if e])
    tmp = new_string.split(' ')
    if not([i for i in tmp if i.count('.') > 1]):
        return new_string
    if new_string.count('.') > 1:
        index = min([i for i, x in enumerate(new_string) if x == '.'])+2
        return new_string[:index]+' ' + new_string[index:]
    return new_string


def validate_number(line):
    """Check if a string is a numeric or is of the form 'number-number'."""
    cond_1 = len(line.split(' ')) == 1
    cond_2 = line.isnumeric()
    cond_3 = all([line.count('-'), line[0].isdigit(), len(line) <= 4])
    return cond_1 and (cond_2 or cond_3)


def find_names(comp, tekst_lin, year, tick):
    """
    Return a list of athletes participating in a comp. with their BIBs.

    Parameters
    ----------
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function (check "database" output for details).
    tekst_lin : list of strings
        A list with parsed PDF file.
    year : integer
        A season when the competition was held
        (details in scraping_fis).
    tick : integer
        similar variable to genre - it says which type
        of competition is considered. Typically it can
        be encoded as:
            0 - World Cup,
            1 - Continental Cup (summer+winter edition),
            2 - Grand Prix/Summer Grand Prix,
            3 - Ski Flying World Championships,
            4 - World Ski Championships,

    Returns
    -------
    lista : list
        A list containing a BIB of every athlete with his/her name,
        which was found in a parsed PDF.
    """
    lista = []
    if tick not in (1, 3):
        names = []
        bibs = []
        if comp['team'] == 1 and int(year) < 2016:
            indexes = [i for i, x in enumerate(tekst_lin) if validate_number(x)]
            lista = [[tekst_lin[i]]+[tekst_lin[i+1]] for i in indexes]
            return lista
        indexes = [i for i, x in enumerate(tekst_lin) if validate(x)]
        check_club = [validate_number(tekst_lin[x-3]) for x in indexes]
        for i, x in enumerate(indexes):
            if int(year) < 2016:
                names.append(tekst_lin[x-3])
                bibs.append(tekst_lin[x-4])
            elif check_club[i]:
                names.append(tekst_lin[x-1])
                bibs.append(tekst_lin[x-3])
            else:
                tmp = tekst_lin[x-1].split(' ')
                names.append(' '.join(tmp[1:]))
                bibs.append(tmp[0])
        lista = [[bibs[i]]+[names[i]] for i, x in enumerate(indexes)]
    else:
        indexes = [i for i, x in enumerate(tekst_lin) if validate(x)]
        check_club = [bool(len(tekst_lin[x-1].split(' '))-1) for x in indexes]
        names = []
        for i, x in enumerate(indexes):
            if check_club[i]:
                names.append(tekst_lin[x-1][:-4])
            else:
                names.append(tekst_lin[x-3])
        lista = [[tekst_lin[x+1]]+[names[i]] for i, x in enumerate(indexes)]
    return lista


def import_start_list(comp, pdf_name, block=False, tekstlin=False):
    """
    Scrap information from FIS start list PDF files.

    Parameters
    ----------
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function (check "database" output for details).
    pdf_name : string
        Unique code for a single competition
        (in the form xxxxJPyyyyzzzz, where xxxx is the year, yyyy is
         a codex of a competition and zzzz is a type of competition, see
         "to_download" section in import_links for details)
    block : True/False, optional
        If True, then the command does not return list of athletes,
        only infos about a competition. The default is False.
    tekstlin : list of strings, optional
        If provided, function does not parse the PDF
        and takes alternative (corrected) version in the same format.
        The default is False.

    Returns
    -------
    lista : list
        list of athletes (output from find_names function)
    comps_infos : Pandas series
        series with the additional information parsed from PDF, like:
            hill size - integer with a HS parameter of a given object, where
            the competition was held,
            k-point - integer with a K point parameter of a given object, where
            the competition was held,
            meter value - float with a value of every meter achieved
            (can be 2, 1.8 or 1.2),
            gate factor - float with a value of every meter of an inrun length,
            wind factor - float with a value of every meter per second of
            front wind,
            training - a True/False variable which indicates whether a comp.
            is from training/trial round or not,
            remaining variables are the same as in "comp" input.
    """
    year = comp['season']
    codex = comp['codex']
    tick = comp['type']
    if 'id' in comp.index:
        name = comp['id'][:10]
    else:
        name = comp['ID']
    if pdf_name.count('RLT') or pdf_name.count('RTRIA'):
        try:
            file_name = name+'SLT.pdf'
            parsed = parser.from_file(os.getcwd()+'\\PDFs\\'+file_name)
        except FileNotFoundError:
            try:
                file_name = name+'SLQ.pdf'
                parsed = parser.from_file(os.getcwd()+'\\PDFs\\'+file_name)
            except FileNotFoundError:
                file_name = name+'SLR1.pdf'
                parsed = parser.from_file(os.getcwd()+'\\PDFs\\'+file_name)
    elif pdf_name.count('RLQ'):
        try:
            file_name = name+'SLT.pdf'
            parsed = parser.from_file(os.getcwd()+'\\PDFs\\'+file_name)
        except FileNotFoundError:
            file_name = name+'SLQ.pdf'
            parsed = parser.from_file(os.getcwd()+'\\PDFs\\'+file_name)
    else:
        file_name = name+'SLR1.pdf'
        parsed = parser.from_file(os.getcwd()+'\\PDFs\\'+file_name)
    tekst = parsed["content"]
    tekst = tekst.replace('* ', '')
    tekst = tekst.replace('*', '')
    tekst = tekst.lower()
    tekst_lin = tekst.splitlines()
    tekst_lin = [i for i in tekst_lin if i]
    if tekstlin:
        tekst_lin = tekstlin
    lista = find_names(comp, tekst_lin, year, tick)
    info = ['season',
            'codex',
            'hill_size',
            'k-point',
            'meter value',
            'gate factor',
            'wind factor',
            'id',
            'training']
    comps_infos = pd.DataFrame([], columns=info)
    word = ['hill size',
            'k-point',
            'meter value',
            'gate factor',
            'wind factor']
    infos = []
    for words in word:
        add = [i for i in tekst_lin if words in i]
        if add:
            infos.append(take_number(add[0]))
        else:
            infos.append(np.nan)
    training = pdf_name.count('RLT') or pdf_name.count('RTRIA')
    new_info = pd.Series([year]+[codex]+infos+[pdf_name]+[training], index=comps_infos.columns)
    comps_infos = comps_infos.append(new_info, ignore_index=True)
    file_name = str(year)+'JP'+str(codex)+'naz.csv'
    if not os.path.isfile(os.getcwd()+'\\nazwy\\'+file_name):
        with open(os.getcwd()+'\\nazwy\\'+file_name, 'w+') as result_file:
            for i, line in enumerate(lista):
                mod_line = ';'.join(line)
                result_file.write(mod_line)
                result_file.write('\n')
        result_file.close()
    with open(os.getcwd()+'\\elastic_nazwy\\'+file_name, 'w+') as result_file:
        for i, line in enumerate(lista):
            mod_line = ';'.join(line)
            result_file.write(mod_line)
            result_file.write('\n')
    result_file.close()
    if not block:
        return([lista, comps_infos])
    return([[], comps_infos])


def zwroc_skoki(comp, tekstlin=False, tekst_import=False, TCS=0):
    """
    Return a list of athletes with all single jumps made in a competition.

    Parameters
    ----------
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function (check "database" output for details).
    tekstlin : list of strings, optional
        If provided, function does not parse the PDF of the competition
        and takes alternative (corrected) version in the same format.
        The default is [].
    tekst_import : list of strings, optional
        If provided, is used as a list of names of the
        consequent athletes. Formatting should be compatible with 
        the import_start_list function. The default is False.
    TCS : integer, optional
        Variable, which determines type of formatting in WC/WSC/SFWC
        competitions. Standard cases are:
            0 (default) - standard formatting,
            1 - formatting to some 4HT competitions (example: 2018JP3059RLQ),
            2 - formatting to some SFWC competitions (example: 2018JP3265RL).

    Returns
    -------
    next_skoki : list
        List of list of strings which contains chunks of parsed PDF lines
        connected with each jumper. Information about each jump is in
        a separate line.
    kwale : integer
        Variable equal to:
            0 - if we have a competition which consists of 2 or more rounds
            or a training/trial round,
            1 - if we have a one-round competition (see 2019JP3090RL
            for instance)
            2 - if we have a qualification round
            (i.e. file name contains 'RLQ')
    team : True/False variable, which indicates team competitions
    TCS : integer, the same as in input
    """
    nazwa = comp['id']+'.pdf'
    kwale = 1
    team = 0
    names_list = []
    if nazwa[-5] == 'Q':
        kwale = 2
    names_list = pd.DataFrame(import_start_list(comp, comp['id']+'.pdf', tekstlin=tekst_import)[0], columns=['bib', 'name'])
    parsed = parser.from_file(os.getcwd()+'\\PDFs\\'+nazwa)
    tekst = parsed["content"]
    tekst = tekst.lower()
    tekst_lin = tekst.splitlines()
    tekst_lin = [i for i in tekst_lin if i]
    if tekstlin:
        tekst_lin = tekstlin
    for line in range(len(tekst_lin[0:8])):
        if tekst_lin[line].count('team') > 0:
            team = 1
    end = []
    word = 'round'
    word2 = 'weather information'
    word2_q = 'prequalified'
    word3 = 'large hill ko'
    for i, line in enumerate(tekst_lin):
        if word2 in line or word2_q in line:
            end.append(i)
        if word in line and i <= 80:
            kwale = 0
        if word3 in line and i <= 80:
            TCS = 1
    if end:
        tekst_lin = tekst_lin[:end[0]]
    lista = [(i, [t for t in names_list['name'] if x.count(t)][0]) for i, x in enumerate(tekst_lin) if any(t for t in names_list['name'] if x.count(t))]+[(len(tekst_lin), 'end')]
    indices = [(lista[i][0], lista[i+1][0], x[1]) for i, x in enumerate(lista[:-1])]
    skoki = [[x]+tekst_lin[s:e] for s, e, x in indices]
    if len(indices) < len(names_list):
        print('Warning: in '+comp['id']+' '+str(len(names_list) - len(indices))+' not found!')
    next_skoki = [conc_numbers(skok, comp, TCS) for i, skok in enumerate(skoki)]
    return([next_skoki, kwale, team, TCS])


def conc_numbers(skok, comp, TCS=0):
    if (comp['type'] == 1 or comp['type'] == 3):
        return conc_numbers_coc(skok, comp, TCS)
    if not comp['training']:
        return skok
    try:
        start = min([i for i, x in enumerate(skok) if x.count('.') and sum([t.isnumeric() for t in x if t.isnumeric()])])
        end = max([i for i, x in enumerate(skok) if min(x.count('.'), sum([t.isnumeric() for t in x if t.isnumeric()]))])
    except ValueError:
        return [skok[0]]
    try:
        end_2 = min([i for i, x in enumerate(skok) if x.count('page')])-1
    except ValueError:
        end_2 = end
    if comp['id'].count('RTRIA'):
        line = ' '.join([skok[start]]+skok[start+2:min(end, end_2)+1])
        if int(comp['season']) >= 2016:
            line = line[4:]
        if skok[start].count('.') == 1:
            line = '0.0 '+line
        return [skok[0], line]
    if comp['id'].count('RLT'):
        no_factor = math.isnan(comp['wind factor'])
        if no_factor:
            shift = 2
        else:
            shift = 4
        indexes = [(i, i+shift) for i in range(start, min(end, end_2)) if not (i-start) % shift]
        lines = [' '.join(skok[i:i+1]+skok[i+2:j]) for i, j in indexes]
        no_speed = [2 - skok[i].count('.') for i, j in indexes]
        lines = [no_speed[i]*'0.0 ' + line for i, line in enumerate(lines)]
        new_lines = [skok[0]]
        for i, line in enumerate(lines):
            tmp = line.split(' ')
            tmp = [x for x in tmp if x == 'dns' or not sum([t.isalpha() for t in x])]
            while tmp[0] == 'dns':
                if comp['training']:
                    new_lines.append(8 * '0.0 ')
                else:
                    new_lines.append(10 * '0.0 ')
                tmp = tmp[1:]
            new_lines.append(' '.join(tmp))
        return new_lines
    return skok


def conc_numbers_coc(skok, comp, TCS=0):
    try:
        start = min([i for i, x in enumerate(skok) if x.count('.') and sum([t.isnumeric() for t in x if t.isnumeric()])])
    except ValueError:
        return [skok[0]]
    ciach_skok = skok[start:]
    try:
        end = start+min([i for i, x in enumerate(ciach_skok) if sum([t.isalpha() for t in x if t.isalpha()])])
    except ValueError:
        end = len(skok)
    if comp['id'].count('RTRIA'):
        if skok[start].count('.') == 1:
            line = skok[start]+'0.0 0.'
        else:
            line = skok[start]
        return [skok[0]]+[' '.join([line]+skok[start+1:end])]
    if comp['training']:
        if TCS == 1:
            if end-start < 7:
                return [skok[0]]+[' '.join(skok[start:end])]
            if skok[end-1].count('.') == 1:
                if skok[end-1][0] == ' ':
                    skok[end-1] = '0.0 '+skok[end-1]
                else:
                    skok[end-1] = skok[end-1]+' 0.0'
            elif end-start == 7:
                skok[end-1] = skok[end-1] + ' 0.0 0.0'
        return [skok[0]]+[' '.join(skok[start:end])]
    if end-start-1:
        pierwszy = [i for i in range(start, end) if not (i-start) % 2]
        drugi = [i for i in range(start, end) if (i-start) % 2 and i != start+3]
        return [skok[0]]+[' '.join([skok[i] for i in pierwszy])]+[' '.join([skok[i] for i in drugi])]
    return [skok[0], skok[start]]


def przeksztalc(comp, string, kwale=0, team=0, TCS=0):
    if comp['type'] in (1, 3) and TCS == 1:
        return przeksztalc_coc_tr(string, comp)
    if comp['type'] in (1, 3):
        return przeksztalc_coc(string, kwale, comp)
    nazwa = comp['id']
    if nazwa.count('RTRIA') or nazwa.count('RLT'):
        return przeksztalc_rlt(comp, string)
    if nazwa.count('RL'):
        return przeksztalc_rl_rlq(comp, string, kwale, team, TCS)
    return []


def przeksztalc_coc_tr(string, comp):
    no_factor = math.isnan(comp['wind factor'])
    sep = 10
    if no_factor:
        sep = 7
    string = rozdziel(string)
    if string.count('.') < sep and string[2] != '.':
        n = [0]
        offset = [-2]
    elif string.count('.') < sep:
        n = []
        offset = []
    else:
        n = [2, 7]
        offset = [-2, 1]
    kropki = [i for i, a in enumerate(string) if a == '.']
    kropki = [kropki[i] for i in n]
    if n:
        nofy_string = [string[0:kropki[0]+offset[0]]]+[string[kropki[i]+offset[i]:kropki[i+1]+offset[i+1]] for i in range(len(kropki)-1)]+[string[kropki[-1]+offset[-1]:]]
        nofy_string = ' '.join(nofy_string)
        wyrazy = nofy_string.split(' ')
        wyrazy = [x for x in wyrazy if x]
    else:
        wyrazy = string.split(' ')
        wyrazy = [x for x in wyrazy if x]
    if string.count('.') < sep and string[2] != '.':
        filter_1 = [1, 2, 0, 7, 5, 6, 8]
        if no_factor:
            filter_1 = [1, 2, 0, 4]
        string_1 = [wyrazy[i] for i in filter_1]
        return [string_1]
    if string.count('.') < sep:
        filter_1 = [0, 1, 2, 4, 5, 6, 7]
        if no_factor:
            filter_1 = [0, 1, 2, 4]
        string_1 = [wyrazy[i] for i in filter_1]
        return [string_1]
    filter_1 = [0, 1, 2, 9, 10, 11, 15, 17]
    filter_2 = [4, 5, 3, 14, 12, 13, 16, 18]
    if no_factor:
        filter_1 = [0, 1, 2, 7]
        filter_2 = [4, 5, 3, 10]
    string_1 = [wyrazy[i] for i in filter_1]
    string_2 = [wyrazy[i] for i in filter_2]
    return [string_1, string_2]


def przeksztalc_coc(string, kwale, comp):
    nq = 0
    tmp = string.split(' ')
    new_string = string
    if comp['id'].count('RTRIA'):
        tmp = [rozdziel(x) for x in tmp]
        tmp = [tmp[0]]+tmp[3:]
        new_string = ' '.join(tmp)
        return new_string
    if comp['id'].count('RLT'):
        tmp = tmp[0:3]+tmp[4:]
        new_string = ' '.join(tmp)
        return new_string
    if not tmp[0].count('.'):
        tmp = tmp[1:]
        nq = 1
    if tmp[0] != rozdziel(tmp[0]):
        tmp_tmp = rozdziel(tmp[0]).split(' ')
        tmp[0] = tmp_tmp[1]+' ' + tmp_tmp[0]
    tmp = [rozdziel(x) for x in tmp]
    new_string = ' '.join(tmp)
    if nq and not kwale:
        tmp = new_string.split(' ')
        tmp = [x for x in tmp if x]
        if len(tmp) == 15:
            tmp = tmp[:13]+[tmp[14]]+[tmp[13]]
        new_string = ' '.join(tmp)
    return new_string


def rozdziel_rl_rlq(comp, kwale, team, TCS):
    no_factor = math.isnan(comp['wind factor'])
    n = [12]
    offset = [1]
    if TCS and kwale == 2 and no_factor:
        n = [0, 10]
        offset = [2, 2]
        return([n, offset])
    if kwale:
        n = []
        offset = []
    if kwale and team:
        n = [12]
        offset = [1]
    if TCS and kwale == 2 and not no_factor:
        n = [12]
        offset = [2]
    if TCS == 2:
        n = [1, 12]
        offset = [2, 1]
    if no_factor:
        if kwale:
            n = [0]
            offset = [2]
        else:
            n = []
            offset = []
    return([n, offset])


def przeksztalc_rl_rlq(comp, string, kwale, team, TCS):
    no_factor = math.isnan(comp['wind factor'])
    string = string.replace('pq', '0.')
    string = string.replace('©', '')
    try:
        limit = max([i for i, x in enumerate(string) if x.isalpha()])
        string = string[limit+2:]
    except ValueError:
        pass
    nowy_string = string.replace('©', '')
    tmp = nowy_string.split(' ')
    if team and tmp[0].count('-'):
        del tmp[0]
    tmp = [x for x in tmp if not sum(i.isalpha() for i in x)]
    nowy_string = ' '.join(tmp)
    pozycja = nowy_string.find('.')+2
    nowy_string = nowy_string[:pozycja]+' '+nowy_string[pozycja:]
    nowy_string = re.sub(r'[a-z]+', '', nowy_string, re.I)
    nowy_string = nowy_string.replace('©', '')
    nowy_string = nowy_string.replace('#', '')
    znacznik = nowy_string.find('*')
    if znacznik:
        nowy_string = nowy_string[(znacznik+1):]
    wyrazy = nowy_string.rsplit(' ', 2)
    nowy_string = wyrazy[1] + ' ' + wyrazy[2] + ' ' + wyrazy[0]
    if TCS == 2:
        string = string.replace('©', '')
        tmp = string.split(' ')
        tmp = [x for x in tmp if x]
        tmp = tmp[-10:] + tmp[:-10]
        nowy_string = ' '.join(tmp)
    if no_factor:
        if comp['id'].count('RLQ'):
            tmp = string.split(' ')
            tmp = [x for x in tmp if x]
            tmp = tmp[1:] + tmp[:1]
            nowy_string = ' '.join(tmp)
        else:
            nowy_string = string
    n, offset = rozdziel_rl_rlq(comp, kwale, team, TCS)
    kropki = [i for i, a in enumerate(nowy_string) if a == '.']
    kropki = [kropki[i] for i in n]
    if n:
        nofy_string = [nowy_string[0:kropki[0]+offset[0]]]+[nowy_string[kropki[i]+offset[i]:kropki[i+1]+offset[i+1]] for i in range(len(kropki)-1)]+[nowy_string[kropki[-1]+offset[-1]:]]
        nofy_string = ' '.join(nofy_string)
        wyrazy = nofy_string.split(' ')
    else:
        nofy_string = nowy_string
    if TCS == 1 and kwale == 2 and not no_factor:
        nofy_string = ' '.join(wyrazy[0:3]) + ' ' + ' '.join(wyrazy[4:13]) + ' ' + ' '.join(wyrazy[14:])
    elif TCS == 1 and kwale == 2 and no_factor:
        nofy_string = ' '.join(wyrazy[0:1]) + ' ' + ' '.join(wyrazy[2:11]) + ' ' + ' '.join(wyrazy[12:])
    return nofy_string


def przeksztalc_rlt(comp, string):
    string = string.replace('©', '')
    nowy_string = string.split()
    nowy_string = [x for x in nowy_string if x]
    if nowy_string.count('dns') and comp['id'].count('rtria'):
        return [0, 0, 0, 0, 0, 0, 0, 0]
    if nowy_string.count('dns') and comp['id'].count('rlt'):
        return [0, 0, 0, 0, 0, 0, 0, 0]
    if comp['id'].count('RTRIA'):
        nofy_string = nowy_string[:2]+nowy_string[-4:]+nowy_string[2:-4]
    else:
        nofy_string = nowy_string[:2]+nowy_string[-2:]+nowy_string[2:-2]
    if not math.isnan(comp['wind factor']):
        return nofy_string+(8-len(nofy_string))*['0.0']
    if comp['id'].count('RTRIA'):
        nofy_string = nowy_string[:3]
        return nofy_string
    try:
        limit = max([i for i, x in enumerate(string) if x.isalpha()])
        tmp_string = string[limit+2:]
    except ValueError:
        tmp_string = string
    nowy_string = tmp_string.split()
    if nowy_string[2].count('.')==2:
        nowy_string = nowy_string[:2] + [nowy_string[2][:4]]
    else:
        nowy_string = nowy_string[:2] + [nowy_string[2][:2]]
    return nowy_string


def column_info(comp, kwale, team):
    no_factor = math.isnan(comp['wind factor'])
    names = ['name',
             'wind',
             'wind_comp',
             'speed',
             'dist',
             'dist_points',
             'note_1',
             'note_2',
             'note_3',
             'note_4',
             'note_5',
             'note_points',
             'points',
             'loc',
             'gate',
             'gate_points']
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    if no_factor:
        indices = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    if kwale and not team:
        indices = [0, 1, 2, 12, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15]
        if no_factor and kwale == 2:
            indices = [0, 12, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14]
        if no_factor and kwale == 1:
            indices = [0, 12, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    nazwa = comp['id']
    if nazwa.count('RTRIA'):
        indices = [0, 3, 4, 1, 2, 5, 13, 14, 15]
        if no_factor:
            indices = [0, 3, 4, 14]
    elif nazwa.count('RLT'):
        indices = [0, 3, 4, 2, 13, 14, 1, 5, 15]
        if no_factor:
            indices = [0, 3, 4, 14]
    if comp['type'] in (1, 3) and not kwale:
        indices = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2, 1, 14, 15]
        if no_factor:
            indices = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    if comp['type'] in (1, 3) and kwale:
        indices = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2, 14, 15]
        if no_factor:
            indices = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    if comp['type'] in (1, 3) and nazwa.count('RLT'):
        indices = [0, 3, 4, 14, 2, 5, 13, 1, 15]
        if no_factor:
            indices = [0, 3, 4, 14, 13]
    if comp['type'] in (1, 3) and nazwa.count('RTRIA'):
        indices = [0, 4, 3, 14, 2, 5, 13, 1, 15]
        if no_factor:
            indices = [0, 4, 3, 14, 13]
    return([names[k] for k in indices])


def znowu_przeksztalc(comp, skok, kwale=0, team=0, TCS=0, show_all=0):
    exit_code = 0
    output = [idx for idx, line in enumerate(skok) if line.count('.') > 4 and sum(x.isdigit() for x in line)]
    output = [x for x in output if x <= 10]
    if TCS == 1 and (comp['type'] in (1,3)):
        if len(skok) > 1:
            skok = [skok[0]]+przeksztalc(comp, skok[1], kwale, team, TCS)
            output = list(range(1,len(skok)))
    if len(output) > 2 and not comp['training']:
        print('Uwaga: zawodnik '+skok[0]+' oddał '+str(len(output))+" skoki!")
    if kwale and len(output) > 1:
        print('Uwaga: zawodnik '+skok[0]+' oddał '+str(len(output))+" skoki w jednoseryjnym konkursie!")
    info = column_info(comp, kwale, team)
    new_jump = pd.DataFrame([], columns=info)
    for line in output:
        name = skok[0]
        if TCS == 1 and (comp['type'] == 1 or comp['type'] == 3):
            notes_pre = skok[line]
        else:
            notes_pre = przeksztalc(comp, skok[line], kwale, team, TCS)
            if not comp['training'] or ((comp['type'] == 1 or comp['type'] == 3) and comp['training']):
                notes_pre = [x for x in notes_pre.split(' ') if x]
        notes = [float(x) for x in notes_pre]
        if show_all:
            try:
                print(notes, notes_pre)
            except ValueError:
                print(notes_pre)
        passed_values = len(info)
        if len(notes) == passed_values - 2:
            notes.append(0)
        data = pd.Series([name]+notes, index=new_jump.columns)
        if not(math.isnan(comp['wind factor'])) and comp['training']:
            conds = [((data['speed'] > 115) or (data['speed'] < 60)) and data['speed'] != 0] + decimal([data['speed'], data['dist_points'], data['dist']], [10, 10, 2])
        else:
            conds = [((data['speed'] > 115) or (data['speed'] < 60)) and data['speed'] != 0] + decimal([data['speed'], data['dist']], [10, 2])
        conds_comp = []
        conds_wind = []
        if not comp['training']:
            conds_comp = [data['note_points'] > 60, data['note_5'] > 20]+decimal([data['points'], data['note_1'], data['note_5'], data['note_points']], [10, 2, 2, 2])
        condition = any(conds)
        if not math.isnan(comp['wind factor']):
            conds_wind = [abs(data['wind']) > 5, abs(data['wind_comp']) > 60] + decimal([data['wind_comp'], data['gate_points'], data['gate']], [10, 10, 1])
        condition = any(conds+conds_comp+conds_wind)
        if condition:
            exit_code = 1
            print(comp['id'])
            print(conds, conds_comp, conds_wind)
            print(data)
        new_jump = new_jump.append(data, ignore_index=True)
    return [new_jump, exit_code]


def collect(comp, tekstlin=False, tekst_start=False, TCS=0, show_all=0):
    jumps, kwale, team, TCS = zwroc_skoki(comp, tekstlin=tekstlin, tekst_import=tekst_start, TCS=TCS)
    exit_code = 0
    info = column_info(comp, kwale, team)
    database = pd.DataFrame([], columns=info)
    for jump in jumps:
        new_jumps, exit_code_tmp = znowu_przeksztalc(comp, jump, kwale, team, TCS, show_all)
        exit_code = exit_code+exit_code_tmp
        database = database.append(new_jumps, ignore_index=True)
    return([database, exit_code])

list_of_files = glob.glob(os.getcwd()+'/comps/*')
comps = max(list_of_files, key=os.path.getctime)
# comps = pd.read_csv(comps)
comps = pd.read_csv(os.getcwd()+'/comps/2011_2012_2013_2014_2015_COC.csv')
comps = comps[comps['k-point'].notnull()]
exit_codes = []
errors = []
for k, comp_to_process in comps.iterrows():
    directory = os.getcwd()+'\\results\\'+comp_to_process['id']+'.csv'
    try:
        content = zwroc_skoki(comp_to_process)
        [dalej, warn] = collect(comp_to_process)
        if (warn or dalej.empty) and not os.path.isfile(directory):
            exit_codes.append(comp_to_process)
            print(comp_to_process)
            continue
        if not warn and not os.path.isfile(directory):
            dalej.to_csv(directory, index=False)
        dalej.to_csv(os.getcwd()+'\\elastic_results\\'+comp_to_process['id']+'.csv', index=False)
    except:
        if not os.path.isfile(directory):
            errors.append(comp_to_process)
            print(comp_to_process)
"""
to_fix = errors

exit_codes = []
errors = []
for comp_to_fix in to_fix:
    print(comp_to_fix)
    file_name = os.getcwd()+'\\results\\'+comp_to_fix['id']+'.csv'
    template = 1
    content = zwroc_skoki(comp_to_fix, TCS=template)
    [dalej, warn] = collect(comp_to_fix, TCS=template)
    old_comp = math.isnan(comp_to_fix['wind factor'])
    if template == 1 and comp_to_fix['type'] in (1, 3) and not old_comp:
        dalej = dalej.drop(['gate_points'], axis=1)
    if (warn or dalej.empty) and not os.path.isfile(file_name):
        exit_codes.append(comp_to_fix)
        print(comp_to_fix)
        continue
    if not warn and not os.path.isfile(file_name):
        dalej.to_csv(file_name, index=False)
    dalej.to_csv(os.getcwd()+'\\elastic_results\\'+comp_to_fix['id']+'.csv', index=False)
    
n = 60
comp_manual = comps.loc[n]
# comp_manual['type'] = 0
template = 1
parsed_manual = parser.from_file(os.getcwd()+'\\PDFs\\'+comp_manual['id']+'.pdf')
tekst_manual = parsed_manual["content"]
tekst_manual = tekst_manual.lower()
tekst_manual = tekst_manual.splitlines()
tekst_manual = [i for i in tekst_manual if i]
try:
    parsed_start = parser.from_file(os.getcwd()+'\\PDFs\\'+comp_manual['id'][:10]+'SLT.pdf')
except FileNotFoundError:
    try:
        parsed_start = parser.from_file(os.getcwd()+'\\PDFs\\'+comp_manual['id'][:10]+'SLQ.pdf')
    except FileNotFoundError:
        parsed_start = parser.from_file(os.getcwd()+'\\PDFs\\'+comp_manual['id'][:10]+'SLR1.pdf')
tekst_start = parsed_start["content"]
tekst_start = tekst_start.lower()
tekst_start = tekst_start.splitlines()
tekst_start = [i for i in tekst_start if i]
content_start = import_start_list(comp_manual, comp_manual['id']+'.pdf', tekstlin=tekst_start)
content = zwroc_skoki(comp_manual, tekst_manual, tekst_start, TCS=template)
dalej, warn = collect(comp_manual, tekst_manual, tekst_start, TCS=template, show_all=True)
old_comp = math.isnan(comp_manual['wind factor'])
if template == 1 and comp_manual['type'] in (1, 3) and not old_comp:
    dalej = dalej.drop(['gate_points'], axis=1)
dalej.to_csv(comp_manual['id']+'.csv', index=False)
"""