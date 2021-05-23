"""
Script to download FIS ski jumping documents.

@author: wrotki8778
"""
import os
import os.path
import time
from datetime import datetime, date
import pandas as pd
import numpy as np
from tika import parser
import requests
from bs4 import BeautifulSoup

os.chdir('C:/Users/kubaf/Documents/Skoki')


def is_number(s):
    """Check that s can be described as a number or not."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def take_number(string):
    """Return a first numeric value in a given string."""
    string = string.replace('m', '')
    string = string.replace('/', ' ')
    tmp = string.split(' ')
    take = [x for x in tmp if is_number(x)]
    try:
        return float(take[0])
    except IndexError:
        return np.nan
    return np.nan


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


def scraping_fis(soup, year):
    """
    Return a list of information from a scrapped FIS site.

    Inputs:
        soup - FIS site of a competition parsed by a BeautifulSoup module,
        year - number (1000-9999) representing a given season,

    Outputs:
        database - a Pandas Series with:
            codex - number (1000-9999) representing a given competition,
            place - string with a city where the competition was held,
            date - list with a date when the competition was held
            when a month, day and a year are written separately,
            gender - 'Men'/'Women' variable saying whether it was a mens' race
            or not,
            hill_size - HS of a hill where the competition was held,
            team - 0/1 variable saying whether it was a team race
            or not,
            year - same as input.
        names_list - a list with names of athletes starting in
        a competition, where we have:
            bib - number connected with a given athlete
            (not provided if team variable is 1),
            fis_code - number representing a given athlete in a FIS database,
            name - full name of an athlete.
        Moreover names_list is exported to a separate folder in a .csv file.
    """
    info = ['codex', 'place', 'month', 'day', 'year',
            'gender', 'hill_size', 'team', 'season']
    database = pd.DataFrame([], columns=info)
    names_all = []
    for tag in soup.find_all('span', {'class': 'event-details__field'}):
        codex = tag.text[-4:]
    for tag in soup.find_all('h1', {'class': 'heading heading_l2 heading_white heading_off-sm-style'}):
        place = tag.text
    for tag in soup.find_all('span', {'class': 'date__full'}):
        date = tag.text
        date = tag.text.replace(',', '').split()
    for tag in soup.find_all('div', {'class': 'event-header__kind'}):
        tmp = tag.text.replace('', '').split()
        gender = tmp[0][:-2]
        hill_size = tmp[-1][2:]
        if len(tmp) >= 3:
            team = 1
        else:
            team = 0
        infos = [codex]+[place]+date+[gender]+[hill_size]+[team]+[year]
        new_comp = pd.Series(infos, index=database.columns)
        database = database.append(new_comp, ignore_index=True)
        names_list = []
        if team:
            class_link = 'table-row table-row_theme_additional'
        else:
            class_link = 'table-row'
        for tag in soup.find_all('a', {'class': class_link}):
            tmp = tag.text.replace('\n', '<')
            tmp = tmp.split('<')
            tmp = [x for x in tmp if x]
            for i, line in enumerate(tmp):
                nazwa = line.split(' ')
                nazwa = [x for x in nazwa if x]
                tmp[i] = ' '.join(nazwa)
                if len(nazwa) > 1:
                    tmp[i+1:] = []
                    break
            if team:
                names_list.append(tmp[-4:])
            else:
                names_list.append(tmp[-3:])
        file_name = str(year)+'JP'+str(codex)+'nazfis.csv'
        if not os.path.isfile(os.getcwd()+'\\nazwy\\'+file_name):
            with open(os.getcwd()+'\\nazwy\\'+file_name, 'w+') as result_file:
                for i, line in enumerate(names_list):
                    mod_line = ';'.join(line)
                    result_file.write(mod_line)
                    result_file.write('\n')
        with open(os.getcwd()+'\\elastic_nazwy\\'+file_name, 'w+') as result_file:
            for i, line in enumerate(names_list):
                mod_line = ';'.join(line)
                result_file.write(mod_line)
                result_file.write('\n')
        result_file.close()
        names_all = names_all+[names_list]
    return([database, names_all])


def import_links(years=[2021], genre='GP',
                 to_download=['RL', 'RLQ',
                              'SLQ', 'SLR1',
                              'RLT', 'RTRIA',
                              'SLT'],
                 import_num=0, scrap=True):
    """
    Return a list of information by scraping FIS sites.

    Inputs:
        years - list of numbers with corresponding seasons
        to scrap (for example number 2019 corresponds to 2018/19 season etc.),
        genre - type of cup/competition to scrap,
        with the following meaning:
            WC - World Cup,
            COC - Continental Cup (summer+winter edition),
            GP - Grand Prix/Summer Grand Prix,
            FC - FIS Cup
            WSC - World Ski Championships,
            SFWC - Ski Flying World Championships,
            WJC - Junior World Ski Championships.
        to_download - type of file to download with the following meaning:
            RL - official competition,
            RLQ - qualification round,
            RLT - training round(s),
            RTRIA - trial round,
            SLR1 - start list before 1st round,
            SLQ - start list before qualification,
            SLT - start list before training round(s),
        import_num - if non-zero, selects only n last weekends
        from the initial FIS search results (accelerates processing),
        scrap - if False, then no information will be downloaded
        from the sites found (except PDFs).
    Outputs:
        grouped_links - contains list of links with consequent weekends
        (competition groups),
        single_links - contains list of links with consequent single
        competitions,
        fis_codes - contains list of unique codes for each competition found
        (in the form xxxxJPyyyy, where xxxx is the year and yyyy is
         a codex of a competition)
        database - a Pandas dataframe, which includes information parsed
        from the website (empty, if scrap is False) by scraping_fis function,
        names_list - separate list containing athletes participating
        in every competition (details in scraping_fis function).
    """
    info = ['codex',
            'place',
            'month',
            'day',
            'year',
            'gender',
            'hill_size',
            'team',
            'season']
    single_links = []
    grouped_links = []
    database = pd.DataFrame([], columns=info)
    names_list = []
    fis_codes = []
    for year in years:
        time.sleep(5)
        url = 'https://www.fis-ski.com/DB/?eventselection=results&place=&sectorcode=JP&seasoncode='+str(year)+'&categorycode='+genre+'&disciplinecode=&gendercode=&racedate=&racecodex=&nationcode=&seasonmonth=X-'+str(year)+'&saveselection=-1&seasonselection='
        r = requests.get(url, headers={'user-agent': 'ejdzent'})
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup.find_all('a', {'class': 'g-sm justify-left hidden-xs hidden-md-up bold'}, href=True):
            grouped_links.append(tag['href'])
    if import_num:
        grouped_links = grouped_links[:import_num]
    for url in grouped_links:
        time.sleep(4)
        year = url[-4:]
        r = requests.get(url, headers={'user-agent': 'ejdzent'})
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup.find_all('a', {'class': 'px-1 g-lg-3 g-md-3 g-sm-4 g-xs-4 justify-left'}, href=True):
            single_links.append([tag['href'], year])
    for item in single_links:
        time.sleep(4)
        url = item[0]
        print(url)
        year = item[1]
        r = requests.get(url, headers={'user-agent': 'ejdzent'})
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup.find_all('span', {'class': 'event-details__field'}):
            codex = tag.text[-4:]
            print(year+'JP'+codex)
        for suffix in to_download:
            tmp_file_name = year+'JP'+codex+suffix
            fis_codes.append(tmp_file_name)
        if scrap:
            tmp, tmp_2 = scraping_fis(soup, year)
            database = database.append(tmp)
            names_list.append(tmp_2)
    for kod in fis_codes:
        data = kod[0:4]
        cc = kod[6:10]
        url = 'http://medias3.fis-ski.com/pdf/'+data+'/JP/'+cc+'/'+kod+'.pdf'
        if not os.path.isfile(os.getcwd()+'\\PDFs\\'+kod+'.pdf'):
            time.sleep(1)
            r = requests.get(url, allow_redirects=True)
        else:
            print('Pominięto konkurs: '+kod+'.pdf')
            continue
        if r.status_code == 404:
            continue
        time.sleep(4)
        open(os.getcwd()+'\\PDFs\\'+kod+'.pdf', 'wb').write(r.content)
        if os.path.getsize(os.getcwd()+'\\PDFs\\'+kod+'.pdf') < 15:
            os.remove(os.getcwd()+'\\PDFs\\'+kod+'.pdf')
        else:
            print('Pobrano konkurs: '+kod+'.pdf')
    return [grouped_links, single_links, fis_codes, database, names_list]


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
        function (check "database" output for details).
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


take_years = [2020]
type_indice = 0
types = ['WC', 'COC', 'GP', 'FC', 'SFWC', 'WSC', 'WJC']
new_data = import_links(years=take_years, genre=types[type_indice],
                        import_num=3)

comps_init = new_data[3]
comps_init['type'] = type_indice
to_process = ['RLQ', 'RL', 'RLT', 'RTRIA', 'RLT2']
to_process = [x+'.pdf' for x in to_process]
list_of_pdfs = os.listdir(os.getcwd()+'\\PDFs\\')
comps_init['ID'] = comps_init.apply(lambda x:
                                    str(x['season'])+'JP'+str(x['codex']),
                                    axis=1).tolist()
list_of_pdfs = [x for x in list_of_pdfs
                if any(t for t in to_process if t in x)
                and any(t for t in comps_init['ID'] if t in x)]
list_of_pdfs.reverse()


start_lists = []
comps_names = ['season', 'codex', 'hill_size', 'k-point', 'meter value',
               'gate factor', 'wind factor', 'id']
competitions = pd.DataFrame([], index=comps_names)

for pdf_file in list_of_pdfs:
    competition = comps_init[comps_init['ID'] == pdf_file[:10]].iloc[0]
    [list_of_names, info_comp] = import_start_list(competition, pdf_file[:-4])
    competitions = competitions.append(info_comp, ignore_index=True)
    start_lists = start_lists+[[list_of_names]]
comps_init = comps_init.drop(['ID'], axis=1)
competitions = pd.merge(competitions, comps_init,
                        on=['season', 'codex'], how='inner')
competitions['date'] = competitions.apply(lambda x:
                                          to_date(x['day'],
                                                  x['month'],
                                                  x['year']),
                                          axis=1)
competitions = competitions.drop(['month', 'day', 'year'], axis=1)
competitions['type'] = type_indice
save_name = '_'.join([str(x) for x in take_years])+'_'\
    + str(types[type_indice])+'_'+str(date.today())+'.csv'
if not os.path.isfile(os.getcwd()+'\\comps\\'+save_name):
    competitions.to_csv(os.getcwd()+'\\comps\\'+save_name, index=False)
competitions.to_csv(os.getcwd()+'\\elastic_comps\\'+save_name, index=False)
