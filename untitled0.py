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


def disperse_text(string):
    """Process a string to split a multi-'.'-substrings."""
    new_string = string
    if string.count('-'):
        new_string = string.split('-')[0]+' '\
            + ' '.join(['-'+e for e in string.split('-')[1:] if e])
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


def find_names(comp, line_text, year, tick):
    """
    Return a list of athletes participating in a comp. with their BIBs.

    Parameters
    ----------
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function (check "database" output for details).
    line_text : list of strings
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
            3 - FIS Cup,
            4 - Ski Flying World Championships,
            5 - World Ski Championships,
            6 - Junior World Ski Championships.

    Returns
    -------
    full_list : list
        A list containing a BIB of every athlete with his/her name,
        which was found in a parsed PDF.
    """
    full_list = []
    line_text = [x.replace(' *', '') for x in line_text]
    if tick not in (1, 3, 6):
        names = []
        bibs = []
        if comp['team'] == 1 and int(year) < 2016:
            indexes = [i for i, x in enumerate(line_text)
                       if validate_number(x)]
            full_list = [[line_text[i]]+[line_text[i+1]] for i in indexes]
            return full_list
        indexes = [i for i, x in enumerate(line_text) if validate(x)]
        check_club = [validate_number(line_text[x-3]) for x in indexes]
        check_club_two = [bool(len(line_text[x-1].split(' '))-1)
                          for x in indexes]
        for i, x in enumerate(indexes):
            if int(year) < 2016:
                if 1-check_club_two[i]:
                    names.append(line_text[x-3])
                    bibs.append(line_text[x-4])
                else:
                    names.append(line_text[x-1][:-4])
                    bibs.append(line_text[x-2])
            elif check_club[i]:
                names.append(line_text[x-1])
                bibs.append(line_text[x-3])
            else:
                print(line_text[x-1])
                names.append(line_text[x-1])
                bibs.append(line_text[x-2])
        full_list = [[bibs[i]]+[names[i]] for i, x in enumerate(indexes)]
    else:
        indexes = [i for i, x in enumerate(line_text) if validate(x)]
        check_club = [bool(len(line_text[x-1].split(' '))-1) for x in indexes]
        names = []
        for i, x in enumerate(indexes):
            if check_club[i]:
                names.append(line_text[x-1][:-4])
            else:
                names.append(line_text[x-3])
        full_list = [[line_text[x+1]]+[names[i]]
                     for i, x in enumerate(indexes)]
    return full_list


def import_start_list(comp, pdf_name, block=False, manual_text=False):
    """
    Scrap information from FIS start list PDF files.

    Parameters
    ----------
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py script (check "database" output for details).
    pdf_name : string
        Unique code for a single competition
        (in the form xxxxJPyyyyzzzz, where xxxx is the year, yyyy is
         a codex of a competition and zzzz is a type of competition, see
         "to_download" section in import_links for details)
    block : True/False, optional
        If True, then the command does not return list of athletes,
        only infos about a competition. The default is False.
    manual_text : list of strings, optional
        If provided, function does not parse the PDF
        and takes alternative (corrected) version in the same format.
        The default is False.

    Returns
    -------
    full_list : list
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
    text = parsed["content"]
    text = text.replace(' *', '')
    text = text.replace('* ', '')
    text = text.replace('*', '')
    text = text.lower()
    line_text = text.splitlines()
    line_text = [i for i in line_text if i]
    if manual_text:
        line_text = manual_text
    full_list = find_names(comp, line_text, year, tick)
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
        add = [i for i in line_text if words in i]
        if add:
            infos.append(take_number(add[0]))
        else:
            infos.append(np.nan)
    training = pdf_name.count('RLT') or pdf_name.count('RTRIA')
    new_info = pd.Series([year]+[codex]+infos+[pdf_name]+[training],
                         index=comps_infos.columns)
    comps_infos = comps_infos.append(new_info, ignore_index=True)
    file_name = str(year)+'JP'+str(codex)+'naz.csv'
    if not os.path.isfile(os.getcwd()+'\\nazwy\\'+file_name):
        with open(os.getcwd()+'\\nazwy\\'+file_name, 'w+') as result_file:
            for i, line in enumerate(full_list):
                mod_line = ';'.join(line)
                result_file.write(mod_line)
                result_file.write('\n')
        result_file.close()
    with open(os.getcwd()+'\\elastic_nazwy\\'+file_name, 'w+') as result_file:
        for i, line in enumerate(full_list):
            mod_line = ';'.join(line)
            result_file.write(mod_line)
            result_file.write('\n')
    result_file.close()
    if not block:
        return([full_list, comps_infos])
    return([[], comps_infos])


def get_jumps(comp, manual_text=False, import_text=False, pdf_format=0):
    """
    Return a list of athletes with all single jumps made in a competition.

    Parameters
    ----------
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py script (check "database" output for details).
    manual_text : list of strings, optional
        If provided, function does not parse the PDF of the competition
        and takes alternative (corrected) version in the same format.
        The default is [].
    import_text : list of strings, optional
        If provided, is used as a list of names of the
        consequent athletes. Formatting should be compatible with
        the import_start_list function. The default is False.
    pdf_format : integer, optional
        Variable, which determines type of formatting in WC/WSC/SFWC
        competitions. Standard cases are:
            0 (default) - standard formatting,
            1 - formatting to some 4HT competitions (example: 2018JP3059RLQ),
            2 - formatting to some SFWC competitions (example: 2018JP3265RL).

    Returns
    -------
    results_list : list
        List of list of strings which contains chunks of parsed PDF lines
        connected with each jumper. Information about each jump is in
        a separate line.
    no_rounds : integer
        Variable equal to:
            0 - if we have a competition which consists of 2 or more rounds
            or a training/trial round,
            1 - if we have a one-round competition (see 2019JP3090RL
            for instance)
            2 - if we have a qualification round
            (i.e. file name contains 'RLQ')
    team : True/False variable, which indicates team competitions
    pdf_format : integer, the same as in input
    """
    comp_name = comp['id']+'.pdf'
    no_rounds = 1
    team = 0
    names_list = []
    if comp_name[-5] == 'Q':
        no_rounds = 2
    names_list = pd.DataFrame(import_start_list(comp, comp['id']+'.pdf',
                                                manual_text=import_text)[0],
                              columns=['bib', 'name'])
    parsed = parser.from_file(os.getcwd()+'\\PDFs\\'+comp_name)
    text = parsed["content"]
    text = text.lower()
    line_text = text.splitlines()
    line_text = [i for i in line_text if i]
    if manual_text:
        line_text = manual_text
    for line in range(len(line_text[0:8])):
        if line_text[line].count('team') > 0:
            team = 1
    end = []
    word = 'round'
    word2 = 'weather information'
    word2_q = 'prequalified'
    word3 = 'large hill ko'
    for i, line in enumerate(line_text):
        if word2 in line or word2_q in line:
            end.append(i)
        if word in line and i <= 80:
            no_rounds = 0
        if word3 in line and i <= 80:
            pdf_format = 1
    if end:
        line_text = line_text[:end[0]]
    full_list = [(i, [t for t in names_list['name'] if x.count(t)][0])
                 for i, x in enumerate(line_text)
                 if any(t for t in names_list['name'] if x.count(t))]\
        + [(len(line_text), 'end')]
    indices = [(full_list[i][0], full_list[i+1][0], x[1])
               for i, x in enumerate(full_list[:-1])]
    jumps = [[x]+line_text[s:e] for s, e, x in indices]
    if len(indices) < len(names_list):
        print('Warning: in '+comp['id']+' '
              + str(len(names_list) - len(indices))+' not found!')
    results_list = [conc_numbers(jump, comp, pdf_format)
                    for i, jump in enumerate(jumps)]
    return([results_list, no_rounds, team, pdf_format])


def conc_numbers(jump, comp, pdf_format=0):
    """
    Concatenate some lines in a jump list to generate a pre-jump-string.

    Parameters
    ----------
    jump : list of strings
        Output of get_jumps procedure, see results_list for details.
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py script (check "database" output for details).
    pdf_format : integer, optional
        Variable, which determines type of formatting in WC/WSC/SFWC
        competitions. Standard cases are:
            0 (default) - standard formatting,
            1 - formatting to some 4HT competitions (example: 2018JP3059RLQ),
            2 - formatting to some SFWC competitions (example: 2018JP3265RL).  

    Returns
    -------
    list of strings
        Concatenated list of lines from the jump variable.

    """
    if comp['type'] in (1, 3, 6):
        return conc_numbers_coc(jump, comp, pdf_format)
    if not comp['training']:
        if math.isnan(comp['gate factor']) and comp['team']:
            if len(jump) > 2:
                jump[2] = ' '.join(jump[1].split(' ')[-3:])+' '+jump[2]
                print(jump[2])
        return jump
    try:
        start = min([i for i, x in enumerate(jump)
                     if x.count('.')
                     and sum([t.isnumeric() for t in x if t.isnumeric()])])
        end = max([i for i, x in enumerate(jump)
                   if min(x.count('.'),
                          sum([t.isnumeric() for t in x if t.isnumeric()]))])
    except ValueError:
        return [jump[0]]
    try:
        end_2 = min([i for i, x in enumerate(jump) if x.count('page')])-1
    except ValueError:
        end_2 = end
    if comp['id'].count('RTRIA'):
        line = ' '.join([jump[start]]+jump[start+2:min(end, end_2)+1])
        if int(comp['season']) >= 2016:
            line = line[4:]
        if jump[start].count('.') == 1:
            line = '0.0 '+line
        return [jump[0], line]
    if comp['id'].count('RLT'):
        no_factor = math.isnan(comp['wind factor'])
        if no_factor:
            shift = 2
            indexes = [(i, i+shift) for i in range(start, min(end, end_2))
                       if not (i-start) % shift]
            lines = [' '.join(jump[i:j]) for i, j in indexes]
        else:
            shift = 4
            indexes = [(i, i+shift) for i in range(start, min(end, end_2))
                       if not (i-start) % shift]
            lines = [' '.join(jump[i:i+1]+jump[i+2:j]) for i, j in indexes]
        no_speed = [2 - jump[i].count('.') for i, j in indexes]
        lines = [no_speed[i]*'0.0 ' + line for i, line in enumerate(lines)]
        new_lines = [jump[0]]
        for i, line in enumerate(lines):
            tmp = line.split(' ')
            tmp = [x for x in tmp if x == 'dns'
                   or not sum([t.isalpha() for t in x])]
            while tmp[0] == 'dns':
                if comp['training']:
                    new_lines.append(8 * '0.0 ')
                else:
                    new_lines.append(10 * '0.0 ')
                tmp = tmp[1:]
            new_lines.append(' '.join(tmp))
        return new_lines
    return jump


def conc_numbers_coc(jump, comp, pdf_format=0):
    """
    Concatenate some lines in a jump list to generate a pre-jump-string.

    Parameters
    ----------
    jump : list of strings
        Output of get_jumps procedure, see results_list for details.
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py script (check "database" output for details).
    pdf_format : integer, optional
        Variable, which determines type of formatting in WC/WSC/SFWC
        competitions. Standard cases are:
            0 (default) - standard formatting,
            1 - formatting to some 4HT competitions (example: 2018JP3059RLQ),
            2 - formatting to some SFWC competitions (example: 2018JP3265RL).

    Returns
    -------
    list of strings
        Concatenated list of lines from the jump variable.

    """
    try:
        start = min([i for i, x in enumerate(jump)
                     if x.count('.')
                     and sum([t.isnumeric() for t in x if t.isnumeric()])])
    except ValueError:
        return [jump[0]]
    cutted_jump = jump[start:]
    try:
        end = start+min([i for i, x in enumerate(cutted_jump)
                         if sum([t.isalpha() for t in x if t.isalpha()])])
    except ValueError:
        end = len(jump)
    if comp['id'].count('RTRIA'):
        if jump[start].count('.') == 1:
            line = jump[start]+'0.0 0.'
        else:
            line = jump[start]
        return [jump[0]]+[' '.join([line]+jump[start+1:end])]
    if comp['training']:
        if pdf_format == 1:
            if end-start < 7:
                return [jump[0]]+[' '.join(jump[start:end])]
            if jump[end-1].count('.') == 1:
                if jump[end-1][0] == ' ':
                    jump[end-1] = '0.0 '+jump[end-1]
                else:
                    jump[end-1] = jump[end-1]+' 0.0'
            elif end-start == 7:
                jump[end-1] = jump[end-1] + ' 0.0 0.0'
        return [jump[0]]+[' '.join(jump[start:end])]
    if end-start-1:
        pierwszy = [i for i in range(start, end) if not (i-start) % 2]
        drugi = [i for i in range(start, end)
                 if (i-start) % 2 and i != start+3]
        return [jump[0]]+[' '.join([jump[i] for i in pierwszy])]\
            + [' '.join([jump[i] for i in drugi])]
    return [jump[0], jump[start]]


def transform(comp, string, no_rounds=0, team=0, pdf_format=0):
    """
    General procedure to generate a suitable jump-string to further processing.

    Parameters
    ----------
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py script (check "database" output for details).
    string : string
        Line with the infos about a given jump.
    no_rounds : integer
        Variable equal to:
            0 - if we have a competition which consists of 2 or more rounds
            or a training/trial round,
            1 - if we have a one-round competition (see 2019JP3090RL
            for instance)
            2 - if we have a qualification round
            (i.e. file name contains 'RLQ')
    team : True/False variable, which indicates team competitions
    pdf_format : integer, optional
        Variable, which determines type of formatting in WC/WSC/SFWC
        competitions. Standard cases are:
            0 (default) - standard formatting,
            1 - formatting to some 4HT competitions (example: 2018JP3059RLQ),
            2 - formatting to some SFWC competitions (example: 2018JP3265RL).

    Returns
    -------
    string or list of strings
        Extracted values from string input.

    """
    if comp['type'] in (1, 3, 6) and pdf_format == 1:
        return transform_coc_training(string, comp)
    if comp['type'] in (1, 3, 6):
        return transform_coc(string, no_rounds, comp)
    comp_name = comp['id']
    if comp_name.count('RTRIA') or comp_name.count('RLT'):
        return transform_rlt(comp, string)
    if comp_name.count('RL'):
        return transform_rl_rlq(comp, string, no_rounds, team, pdf_format)
    return []


def transform_coc_training(string, comp):
    """
    Process a jump-string from the COC rounds with 2 training rounds in a PDF.

    Parameters
    ----------
    string : string
        Line with the infos about a given jump.
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py script (check "database" output for details).

    Returns
    -------
    list
        Extracted values from string input.

    """
    no_factor = math.isnan(comp['wind factor'])
    sep = 10
    if no_factor:
        sep = 7
    string = disperse_text(string)
    if string.count('.') < sep and string[2] != '.':
        placement = [0]
        offset = [-2]
    elif string.count('.') < sep:
        placement = []
        offset = []
    else:
        placement = [2, 7]
        offset = [-2, 1]
    kropki = [i for i, a in enumerate(string) if a == '.']
    kropki = [kropki[i] for i in placement]
    if placement:
        tmp_new_string = [string[0:kropki[0]+offset[0]]]\
            + [string[kropki[i]+offset[i]:kropki[i+1]+offset[i+1]]
               for i in range(len(kropki)-1)]+[string[kropki[-1]+offset[-1]:]]
        tmp_new_string = ' '.join(tmp_new_string)
        parts = tmp_new_string.split(' ')
        parts = [x for x in parts if x]
    else:
        parts = string.split(' ')
        parts = [x for x in parts if x]
    if string.count('.') < sep and string[2] != '.':
        filter_1 = [1, 2, 0, 7, 5, 6, 8]
        if no_factor:
            filter_1 = [1, 2, 0, 4]
        string_1 = [parts[i] for i in filter_1]
        return [string_1]
    if string.count('.') < sep:
        filter_1 = [0, 1, 2, 4, 5, 6, 7]
        if no_factor:
            filter_1 = [0, 1, 2, 4]
        string_1 = [parts[i] for i in filter_1]
        return [string_1]
    filter_1 = [0, 1, 2, 9, 10, 11, 15, 17]
    filter_2 = [4, 5, 3, 14, 12, 13, 16, 18]
    if no_factor:
        filter_1 = [0, 1, 2, 7]
        filter_2 = [4, 5, 3, 10]
    string_1 = [parts[i] for i in filter_1]
    string_2 = [parts[i] for i in filter_2]
    return [string_1, string_2]


def transform_coc(string, no_rounds, comp):
    """
    Process a jump-string from the COC rounds (training + official rounds).

    Parameters
    ----------
    string : string
        Line with the infos about a given jump.
    no_rounds : integer
        Variable equal to:
            0 - if we have a competition which consists of 2 or more rounds
            or a training/trial round,
            1 - if we have a one-round competition (see 2019JP3090RL
            for instance)
            2 - if we have a qualification round
            (i.e. file name contains 'RLQ')
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py script (check "database" output for details).

    Returns
    -------
    new_string : string
        Extracted values from string input.

    """
    not_qual = 0
    team_not_qual = 0
    tmp = string.split(' ')
    new_string = string
    if comp['id'].count('RTRIA'):
        tmp = [disperse_text(x) for x in tmp]
        tmp = [tmp[0]]+tmp[3:]
        new_string = ' '.join(tmp)
        return new_string
    if comp['id'].count('RLT'):
        tmp = tmp[0:3]+tmp[4:]
        new_string = ' '.join(tmp)
        return new_string
    if not tmp[0].count('.'):
        tmp = tmp[1:]
        not_qual = 1
    if tmp[0] != disperse_text(tmp[0]):
        tmp_tmp = disperse_text(tmp[0]).split(' ')
        tmp[0] = tmp_tmp[1]+' ' + tmp_tmp[0]
        team_not_qual = 1
    tmp = [disperse_text(x) for x in tmp]
    new_string = ' '.join(tmp)
    if not_qual and not no_rounds:
        tmp = new_string.split(' ')
        tmp = [x for x in tmp if x]
        if len(tmp) == 15:
            tmp = tmp[:13]+[tmp[14]]+[tmp[13]]
        new_string = ' '.join(tmp)
    if comp['team']:
        tmp = new_string.split(' ')
        tmp = [x for x in tmp if x]
        if not_qual and team_not_qual and len(tmp) == 15:
            tmp = tmp[:-3] + tmp[-2:] + [tmp[-3]]
        if not(not_qual) and team_not_qual:
            tmp = tmp[:11] + [tmp[-1]] + [tmp[11]] + [tmp[-2]] + tmp[12:-2]
        if not(not_qual) and not(team_not_qual):
            tmp = tmp[:11] + [tmp[-2]] + [tmp[11]] + [tmp[-1]] + tmp[12:-2]
        new_string = ' '.join(tmp)
        print(tmp)
    return new_string


def disperse_rl_rlq(comp, no_rounds, team, pdf_format):
    """
    Return an instruction which dots in the string should be shifted.

    Parameters
    ----------
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py script (check "database" output for details).
    no_rounds : integer
        Variable equal to:
            0 - if we have a competition which consists of 2 or more rounds
            or a training/trial round,
            1 - if we have a one-round competition (see 2019JP3090RL
            for instance)
            2 - if we have a qualification round
            (i.e. file name contains 'RLQ')
    team : True/False variable, which indicates team competitions
    pdf_format : integer, optional
        Variable, which determines type of formatting in WC/WSC/SFWC
        competitions. Standard cases are:
            0 (default) - standard formatting,
            1 - formatting to some 4HT competitions (example: 2018JP3059RLQ),
            2 - formatting to some SFWC competitions (example: 2018JP3265RL).

    Returns
    -------
    List of dots to shift (placement) and the number of spaces to add (offset).

    """
    no_factor = math.isnan(comp['wind factor'])
    placement = [12]
    offset = [1]
    if pdf_format and no_rounds == 2 and no_factor:
        placement = [0, 10]
        offset = [2, 2]
        return([placement, offset])
    if no_rounds:
        placement = []
        offset = []
    if no_rounds and team:
        placement = [12]
        offset = [1]
    if pdf_format and no_rounds == 2 and not no_factor:
        placement = [12]
        offset = [2]
    if pdf_format == 2:
        placement = [1, 12]
        offset = [2, 1]
    if no_factor:
        if no_rounds:
            placement = [0]
            offset = [2]
        else:
            placement = []
            offset = []
    return([placement, offset])


def transform_rl_rlq(comp, string, no_rounds, team, pdf_format):
    """
    Process a jump-string from the WC official rounds.

    Parameters
    ----------
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py script (check "database" output for details).
    string : string
        Line with the infos about a given jump
    no_rounds : integer
        Variable equal to:
            0 - if we have a competition which consists of 2 or more rounds
            or a training/trial round,
            1 - if we have a one-round competition (see 2019JP3090RL
            for instance)
            2 - if we have a qualification round
            (i.e. file name contains 'RLQ')
    team : True/False variable, which indicates team competitions
    pdf_format : integer, optional
        Variable, which determines type of formatting in WC/WSC/SFWC
        competitions. Standard cases are:
            0 (default) - standard formatting,
            1 - formatting to some 4HT competitions (example: 2018JP3059RLQ),
            2 - formatting to some SFWC competitions (example: 2018JP3265RL).

    Returns
    -------
    tmp_new_string : string
        Extracted values from string input.

    """
    no_factor = math.isnan(comp['wind factor'])
    string = string.replace('pq', '0.')
    string = string.replace('©', '')
    try:
        limit = max([i for i, x in enumerate(string) if x.isalpha()])
        string = string[limit+2:]
    except ValueError:
        pass
    new_string = string.replace('©', '')
    tmp = new_string.split(' ')
    if team and tmp[0].count('-'):
        del tmp[0]
    tmp = [x for x in tmp if not sum(i.isalpha() for i in x)]
    new_string = ' '.join(tmp)
    pozycja = new_string.find('.')+2
    new_string = new_string[:pozycja]+' '+new_string[pozycja:]
    new_string = re.sub(r'[a-z]+', '', new_string, re.I)
    new_string = new_string.replace('©', '')
    new_string = new_string.replace('#', '')
    star_pointer = new_string.find('*')
    if star_pointer:
        new_string = new_string[(star_pointer+1):]
    parts = new_string.rsplit(' ', 2)
    new_string = parts[1] + ' ' + parts[2] + ' ' + parts[0]
    if pdf_format == 2:
        string = string.replace('©', '')
        tmp = string.split(' ')
        tmp = [x for x in tmp if x]
        tmp = tmp[-10:] + tmp[:-10]
        new_string = ' '.join(tmp)
    if no_factor:
        if comp['id'].count('RLQ'):
            tmp = string.split(' ')
            tmp = [x for x in tmp if x]
            tmp = tmp[1:] + tmp[:1]
            new_string = ' '.join(tmp)
        else:
            new_string = string
    placement, offset = disperse_rl_rlq(comp, no_rounds, team, pdf_format)
    kropki = [i for i, a in enumerate(new_string) if a == '.']
    kropki = [kropki[i] for i in placement]
    if placement:
        tmp_new_string = [new_string[0:kropki[0]+offset[0]]]\
            + [new_string[kropki[i]+offset[i]:kropki[i+1]+offset[i+1]]
               for i in range(len(kropki)-1)]\
                + [new_string[kropki[-1]+offset[-1]:]]
        tmp_new_string = ' '.join(tmp_new_string)
        parts = tmp_new_string.split(' ')
    else:
        tmp_new_string = new_string
    if pdf_format == 1 and no_rounds == 2 and not no_factor:
        tmp_new_string = ' '.join(parts[0:3]) + ' ' \
            + ' '.join(parts[4:13]) + ' ' + ' '.join(parts[14:])
    elif pdf_format == 1 and no_rounds == 2 and no_factor:
        tmp_new_string = ' '.join(parts[0:1]) + ' ' \
            + ' '.join(parts[2:11]) + ' ' + ' '.join(parts[12:])
    return tmp_new_string


def transform_rlt(comp, string):
    """
    Process a jump-string from the WC training/trial round.

    Parameters
    ----------
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py script (check "database" output for details).
    string : string
        Line with the infos about a given jump

    Returns
    -------
    list
        Extracted values from string input.

    """
    string = string.replace('©', '')
    new_string = string.split()
    new_string = [x for x in new_string if x]
    if new_string.count('dns') and comp['id'].count('rtria'):
        return [0, 0, 0, 0, 0, 0, 0, 0]
    if new_string.count('dns') and comp['id'].count('rlt'):
        return [0, 0, 0, 0, 0, 0, 0, 0]
    if comp['id'].count('RTRIA'):
        tmp_new_string = new_string[:2]+new_string[-4:]+new_string[2:-4]
    else:
        tmp_new_string = new_string[:2]+new_string[-2:]+new_string[2:-2]
    if not math.isnan(comp['wind factor']):
        return tmp_new_string+(8-len(tmp_new_string))*['0.0']
    if comp['id'].count('RTRIA'):
        tmp_new_string = new_string[:3]
        return tmp_new_string
    try:
        limit = max([i for i, x in enumerate(string) if x.isalpha()])
        tmp_string = string[limit+2:]
    except ValueError:
        tmp_string = string
    new_string = tmp_string.split()
    if new_string[2].count('.') == 2:
        new_string = new_string[:2] + [new_string[2][:4]]
    else:
        new_string = new_string[:2] + [new_string[2][:2]]
    return new_string


def column_info(comp, no_rounds, team):
    """
    Return the names of variables extracted from transform function.

    Parameters
    ----------
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py script (check "database" output for details).
    no_rounds : integer
        Variable equal to:
            0 - if we have a competition which consists of 2 or more rounds
            or a training/trial round,
            1 - if we have a one-round competition (see 2019JP3090RL
            for instance)
            2 - if we have a qualification round
            (i.e. file name contains 'RLQ')
    team : True/False variable, which indicates team competitions

    Returns
    -------
    list of strings
       List of tags according to subsequent numbers in output of transform
       function.

    """
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
    if no_rounds and not team:
        indices = [0, 1, 2, 12, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15]
        if no_factor and no_rounds == 2:
            indices = [0, 12, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14]
        if no_factor and no_rounds == 1:
            indices = [0, 12, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    comp_name = comp['id']
    if comp_name.count('RTRIA'):
        indices = [0, 3, 4, 1, 2, 5, 13, 14, 15]
        if no_factor:
            indices = [0, 3, 4, 14]
    elif comp_name.count('RLT'):
        indices = [0, 3, 4, 2, 13, 14, 1, 5, 15]
        if no_factor:
            indices = [0, 3, 4, 14]
    if comp['type'] in (1, 3, 6) and not no_rounds:
        indices = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2, 1, 14, 15]
        if no_factor:
            indices = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    if comp['type'] in (1, 3, 6) and no_rounds:
        indices = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2, 14, 15]
        if no_factor:
            indices = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    if comp['type'] in (1, 3, 6) and comp_name.count('RLT'):
        indices = [0, 3, 4, 14, 2, 5, 13, 1, 15]
        if no_factor:
            indices = [0, 3, 4, 14, 13]
    if comp['type'] in (1, 3, 6) and comp_name.count('RTRIA'):
        indices = [0, 4, 3, 14, 2, 5, 13, 1, 15]
        if no_factor:
            indices = [0, 4, 3, 14, 13]
    if comp['type'] in (1, 3, 6) and comp['team']:
        indices = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 14, 2, 15]
    return([names[k] for k in indices])


def further_transform(comp, jump, no_rounds=0,
                      team=0, pdf_format=0, show_all=0):
    """
    Return a dataframe with all jumps of an athlete from the jump string.

    Parameters
    ----------
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py script (check "database" output for details).
    jump : list of strings
        Output of get_jumps procedure, see results_list for details.
    no_rounds : integer, optional
        Variable equal to:
            0 - if we have a competition which consists of 2 or more rounds
            or a training/trial round,
            1 - if we have a one-round competition (see 2019JP3090RL
            for instance)
            2 - if we have a qualification round
            (i.e. file name contains 'RLQ')
        The default is 0.
    team : Boolean, optional
        True/False variable, which indicates team competitions.
        The default is 0.
    pdf_format : integer, optional
        Variable, which determines type of formatting in WC/WSC/SFWC
        competitions. Standard cases are:
            0 (default) - standard formatting,
            1 - formatting to some 4HT competitions (example: 2018JP3059RLQ),
            2 - formatting to some SFWC competitions (example: 2018JP3265RL).
    show_all : Boolean, optional
        If non-zero, then some additional intermediate processed data
        are printed. The default is 0.

    Returns
    -------
    list
        new_jump: Pandas dataframe
            Dataframe consisting of jumps of a given athlete.
        exit_come: integer
            If non-zero, then it produces an alert -- probably data are
            parsed incorrectly. It can be overcomed by manual procedure
            at the end of this script.

    """
    exit_code = 0
    output = [idx for idx, line in enumerate(jump)
              if line.count('.') > 4 and sum(x.isdigit() for x in line)]
    output = [x for x in output if x <= 10]
    if pdf_format == 1 and (comp['type'] in (1, 3, 6)):
        if len(jump) > 1:
            jump = [jump[0]]+transform(comp, jump[1], no_rounds,
                                       team, pdf_format)
            output = list(range(1, len(jump)))
    if len(output) > 2 and not comp['training']:
        print('Alert: jumper '+jump[0]+' jumped '+str(len(output))+" times!")
    if no_rounds and len(output) > 1:
        print('Alert: jumper '+jump[0]+' jumped '
              + str(len(output))+" times in one-round competition!")
    info = column_info(comp, no_rounds, team)
    new_jump = pd.DataFrame([], columns=info)
    for line in output:
        name = jump[0]
        if pdf_format == 1 and comp['type'] in (1, 3, 6):
            notes_pre = jump[line]
        else:
            notes_pre = transform(comp, jump[line], no_rounds,
                                  team, pdf_format)
            if not comp['training'] or (comp['type'] in (1, 3, 6)
                                        and comp['training']):
                notes_pre = [x for x in notes_pre.split(' ') if x]
        if show_all:
            print(notes_pre)
        notes = [float(x) for x in notes_pre]
        if show_all:
            print(notes)
        passed_values = len(info)
        if len(notes) == passed_values - 2:
            notes.append(0)
        data = pd.Series([name]+notes, index=new_jump.columns)
        if not(math.isnan(comp['wind factor'])) and comp['training']:
            conds = [((data['speed'] > 115) or (data['speed'] < 60))
                     and data['speed'] != 0]\
                + decimal([data['speed'], data['dist_points'], data['dist']],
                          [10, 10, 2])
        else:
            conds = [((data['speed'] > 115) or (data['speed'] < 60))
                     and data['speed'] != 0]\
                + decimal([data['speed'], data['dist']], [10, 2])
        conds_comp = []
        conds_wind = []
        if not comp['training']:
            conds_comp = [data['note_points'] > 60, data['note_5'] > 20]\
                + decimal([data['points'], data['note_1'], data['note_5'],
                           data['note_points']], [10, 2, 2, 2])
        condition = any(conds)
        if not math.isnan(comp['wind factor']):
            conds_wind = [abs(data['wind']) > 5, abs(data['wind_comp']) > 60]\
                + decimal([data['wind_comp'], data['gate_points'],
                           data['gate']], [10, 10, 1])
        condition = any(conds+conds_comp+conds_wind)
        if condition:
            exit_code = 1
            print(comp['id'])
            print(conds, conds_comp, conds_wind)
            print(data)
        new_jump = new_jump.append(data, ignore_index=True)
    return [new_jump, exit_code]


def collect(comp, manual_text=False, start_text=False,
            pdf_format=0, show_all=0):
    """
    Process all rows from the parsed PDF into a target dataframe.

    Parameters
    ----------
    comp : Pandas series
        Infos about competition gathered in a way provided by import_links
        function in untitled6.py script (check "database" output for details)
    manual_text : list of strings, optional
        If provided, function does not parse the PDF
        and takes alternative (corrected) version in the same format.
        The default is False.
    start_text : list of strings, optional
        If provided, function does not parse the PDF with the start list file
        and takes alternative (corrected) version in the same format.
        The default is False.
    pdf_format : integer, optional
        Variable, which determines type of formatting in WC/WSC/SFWC
        competitions. Standard cases are:
            0 (default) - standard formatting,
            1 - formatting to some 4HT competitions (example: 2018JP3059RLQ),
            2 - formatting to some SFWC competitions (example: 2018JP3265RL).
    show_all : Boolean, optional
        If non-zero, then some additional intermediate processed data
        are printed. The default is 0.

    Returns
    -------
    list
        new_jump: Pandas dataframe
            Dataframe consisting of jumps of all athletes
            in a given competition.
        exit_come: integer
            If non-zero, then it produces an alert -- probably data are
            parsed incorrectly. It can be overcomed by manual procedure
            at the end of this script.

    """
    jumps, no_rounds, team, pdf_format = get_jumps(comp,
                                                   manual_text=manual_text,
                                                   import_text=start_text,
                                                   pdf_format=pdf_format)
    exit_code = 0
    info = column_info(comp, no_rounds, team)
    database = pd.DataFrame([], columns=info)
    for jump in jumps:
        new_jumps, exit_code_tmp = further_transform(comp, jump, no_rounds,
                                                     team, pdf_format,
                                                     show_all)
        exit_code = exit_code+exit_code_tmp
        database = database.append(new_jumps, ignore_index=True)
    return([database, exit_code])


list_of_files = glob.glob(os.getcwd()+'/comps/*')
# directory = max(list_of_files, key=os.path.getctime)
directory = os.getcwd()+'/comps/2021_FC_2021-01-11.csv'
comps = pd.read_csv(directory)
comps = comps[comps['k-point'].notnull()]
"""
# Standard procedure
exit_codes = []
errors = []

for k, comp_to_process in comps.iterrows():
    directory_res = os.getcwd()+'\\results\\'+comp_to_process['id']+'.csv'
    if os.path.isfile(directory_res):
        continue
    try:
        content = get_jumps(comp_to_process)
        [results, warn] = collect(comp_to_process)
        if (warn or results.empty) and not os.path.isfile(directory_res):
            exit_codes.append(comp_to_process)
            print(comp_to_process)
            continue
        if not warn and not os.path.isfile(directory_res):
            results.to_csv(directory_res, index=False)
        results.to_csv(os.getcwd()+'\\elastic_results\\'
                       + comp_to_process['id']+'.csv', index=False)
    except:
        if not os.path.isfile(directory_res):
            errors.append(comp_to_process)
            print(comp_to_process)

# Procedure to parse some COC training rounds (do not run if unnecessary)

to_fix = errors

exit_codes = []
errors = []
for comp_to_fix in to_fix:
    print(comp_to_fix)
    file_name = os.getcwd()+'\\results\\'+comp_to_fix['id']+'.csv'
    if os.path.isfile(file_name):
        continue
    template = 1
    content = get_jumps(comp_to_fix, pdf_format=template)
    [results, warn] = collect(comp_to_fix, pdf_format=template)
    old_comp = math.isnan(comp_to_fix['wind factor'])
    if template == 1 and comp_to_fix['type'] in (1, 3, 6) and not old_comp:
        results = results.drop(['gate_points'], axis=1)
    if (warn or results.empty) and not os.path.isfile(file_name):
        exit_codes.append(comp_to_fix)
        print(comp_to_fix)
        continue
    if not warn and not os.path.isfile(file_name):
        results.to_csv(file_name, index=False)
    results.to_csv(os.getcwd()+'\\elastic_results\\'+comp_to_fix['id']+'.csv', index=False)
"""
# Procedure to parse competitions manually one by one
# (do not run if unnecessary)

n = 17
comp_manual = comps.loc[n]
# comp_manual['type'] = 0
template = 0
parsed_manual = parser.from_file(os.getcwd()+'\\PDFs\\'+comp_manual['id']+'.pdf')
text_manual = parsed_manual["content"]
text_manual = text_manual.lower()
text_manual = text_manual.splitlines()
text_manual = [i for i in text_manual if i]
try:
    parsed_start = parser.from_file(os.getcwd()+'\\PDFs\\'+comp_manual['id'][:10]+'SLT.pdf')
except FileNotFoundError:
    try:
        parsed_start = parser.from_file(os.getcwd()+'\\PDFs\\'+comp_manual['id'][:10]+'SLQ.pdf')
    except FileNotFoundError:
        parsed_start = parser.from_file(os.getcwd()+'\\PDFs\\'+comp_manual['id'][:10]+'SLR1.pdf')
start_text = parsed_start["content"]
start_text = start_text.lower()
start_text = start_text.splitlines()
start_text = [i for i in start_text if i]
content_start = import_start_list(comp_manual, comp_manual['id']+'.pdf', manual_text=start_text)
content = get_jumps(comp_manual, text_manual, start_text, pdf_format=template)
results, warn = collect(comp_manual, text_manual, start_text, pdf_format=template, show_all=True)
old_comp = math.isnan(comp_manual['wind factor'])
if template == 1 and comp_manual['type'] in (1, 3, 6) and not old_comp:
    results = results.drop(['gate_points'], axis=1)
results.to_csv(comp_manual['id']+'.csv', index=False)
