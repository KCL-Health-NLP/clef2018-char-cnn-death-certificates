# coding: utf-8
''' 
Functions to search for icd codes (or any dictionary entries from file) in text - using ngrams and set intersection
Prepared for CLEF eHealth 2018 Task 1: multilingual ICD coding
Written by Sumithra Velupillai and Natalia Viani April-May 2018
'''

import pandas as pd
import time
import datetime
import itertools
import sys
from collections import defaultdict
from nltk.util import ngrams
from nltk.corpus import stopwords
import nltk
import logging
import sys
import unicodedata
import os



###create dictionary with frequencies###
def create_dic(textf, encoding='utf8'):
#    textf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/corpus/AlignedCauses_2006-2012full.csv'
    #textf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/raw/corpus/CausesCalculees_FR_2006-2012.csv'
    df_training = pd.read_csv(textf, sep=';', error_bad_lines=False, warn_bad_lines=True, encoding=encoding) 
    df_training = df_training.fillna("NULL")
    df_training_icds = df_training.groupby("ICD10").size().reset_index(name="counts")
    df_training_icds = df_training_icds.sort_values(by=["counts"], ascending = False)    
    #dictionary with codes from training (ordered by counts)
    dic={}
    pos=1
    for index, row in df_training_icds.iterrows():
        code = row["ICD10"]
        dic[code] = pos
        pos = pos+1
    return dic

def get_string_ngrams(string, stemming=False, ignorediacritics=False, removestopwords=False, lang='french'):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    log = logging.getLogger('get_string_ngrams')
    all_strings = []
    try:
        string = string.lower().strip()
    except:
        log.error('Error with string: '+str(string))
        string = ''
    ## special handling of ac/fa
    string = string.replace('ac/fa','acfa')
    if ignorediacritics:
        ## ignore diacritics
        try:
            string = ''.join((c for c in unicodedata.normalize('NFD', string) if unicodedata.category(c) != 'Mn'))
        except:
            log.error('Problem with string (ignorediacritics): '+string)
    
#    splitstring = nltk.word_tokenize(string)
    splitstring = nltk.wordpunct_tokenize(string)
    ## removing stopwords
    if removestopwords:
        try:
            splitstring = [s for s in splitstring if s not in stopwords.words(lang)]
        except:
            log.error('Unsupported language provided, skipping remove stopwords')


    if stemming:
        try:
            if lang=='french':
                snowball_stemmer = nltk.stem.snowball.FrenchStemmer()
            elif lang == 'italian':
                snowball_stemmer = nltk.stem.snowball.ItalianStemmer()
            splitstring = [snowball_stemmer.stem(w) for w in splitstring]
        except:
            log.error('Unsupported language provided, skipping stemming')
        
    #print(splitstring)
    ## find all ngrams 1-5 in length
    for i in range(1,5):
        tmp = [' '.join(n) for n in ngrams(splitstring,i)]
        if len(tmp)>0:
            all_strings.append(tmp)
    all_strings = list(itertools.chain.from_iterable(all_strings))
    all_strings = list(set(all_strings))     
    return all_strings

def get_code_string_dict(codedataframe, stemming=False, ignorediacritics=False, removestopwords=False, lang='french'):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    log = logging.getLogger('get_code_string_dict')
    icdcodes = defaultdict()

    for x in range(len(codedataframe)):
        if lang=='french':
            currentid = codedataframe.iloc[x,1]
        else:
            currentid = codedataframe.iloc[x,2]
        currentvalue = codedataframe.iloc[x,0]
        currentvalue = currentvalue.lower().strip()
        if ignorediacritics:
            ## ignore diacritics
            currentvalue = ''.join((c for c in unicodedata.normalize('NFD', currentvalue) if unicodedata.category(c) != 'Mn'))        
        if stemming:
            try:
                if lang=='french':
                    snowball_stemmer = nltk.stem.snowball.FrenchStemmer()
                elif lang=='italian':
                    snowball_stemmer = nltk.stem.snowball.ItalianStemmer()
            #splitstring = nltk.word_tokenize(currentvalue)
                splitstring = nltk.wordpunct_tokenize(currentvalue)
                splitstring = [snowball_stemmer.stem(w) for w in splitstring]
            except:
                log.error('Unsupported language provided, skipping stemming')
            if removestopwords:
                try:
                    splitstring = [s for s in splitstring if s not in stopwords.words(lang)]
                except:
                    log.error('Unsupported language provided, skipping remove stopwords')
            currentvalue = ' '.join(splitstring)
        #print(' '.join(splitstring))
        icdcodes.setdefault(currentid, [])
        ## ignore short terms and acronyms for now
        #if len(currentvalue.lower().strip())>0:
        icdcodes[currentid].append(currentvalue)
    return icdcodes

def find_number_of_lines_expected_in_aligned(df):
    dociddict = defaultdict()
    docidtextdict = defaultdict()
    docidalldict = defaultdict()

    docids = df['DocID'].tolist()
    lineids = df['LineID'].tolist()
    rawstrings = df['RawText'].tolist()
    count = 0
    for d in docids:
        lineid = lineids[count]
        dictid = str(d)+'_'+str(lineid)
        dociddict.setdefault(dictid, 0)
        docidtextdict.setdefault(dictid,'')
        docidalldict.setdefault(dictid,[])
        dociddict[dictid]+=1
        docidtextdict[dictid]=rawstrings[count]
        docidalldict[dictid].append(df.iloc[count].tolist())
        count+=1
    return dociddict, docidtextdict, docidalldict 

def find_codes_in_text_aligned(df, icdcodes_strings,trainingfileforfreq,stemming=True, ignorediacritics=True, removestopwords=True, lang='french', training=True):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    log = logging.getLogger('find_codes_in_text_aligned')
    log.info('searching for codes in text')
    ## list to save all codes and put back in dataframe
    codes_to_add = []
    ## list also with strings associated with a code and found in a rawstring
    full_codes_to_add = []
    new_lines_for_dataframe = []
    training_code_freq_list = create_dic(trainingfileforfreq)
    count = 0
    t0 = time.time()
    rawstrings = df['RawText'].tolist()
    docids = df['DocID'].tolist()
    lineids = df['LineID'].tolist()
    dociddict, docidtextdict, docidalldict = find_number_of_lines_expected_in_aligned(df)
    #goldicds = df['ICD10'].tolist()
#    print('number of doc_line_ids: '+str(len(dociddict)))
    ## loop over unique doc+line ids and get text from that
    for doclineid in sorted(dociddict):
        if ((count % 1000) == 0):
            t1 = time.time()
            log.info('took: '+(str(datetime.timedelta(seconds=(t1-t0)))))
            print (str(count)+' out of: '+str(len(rawstrings)))
            t0 = time.time()
        count+=1
        codes = []
        found_codes = []
#        print(doclineid)
        rr = docidtextdict[doclineid]
        rr = rr.lower().strip()
#        print(rr)
        all_info = docidalldict[doclineid]
        no_lines = dociddict[doclineid]
#        print('no lines: '+str(no_lines))
#        print(all_info)
        try:
            string_p = get_string_ngrams(rr, stemming, ignorediacritics, removestopwords, lang)
        except:
            log.error('Error with line: '+str(count)+' ('+rr+')')
        for i in icdcodes_strings:
            codestrings = icdcodes_strings[i]
            intersect = (set(string_p) & set(codestrings))
            if len(intersect)>0:
                identical = False
                for m in intersect:
                    if rr == m:
                        found_codes.append([m])
                        codes.append(i)
                        identical=True
                        break
                if not identical:
                    found_codes.append(list(intersect))
                    codes.append(i)
#        print('no of codes found: '+str(len(found_codes)))
#        print('found strings in icdcodes: '+str(found_codes))
#        print('found codes: '+str(codes))
        codes_to_save = []
        strings_to_save = []
        if len(codes)==no_lines:
#            print('same number of codes found as in gold')
            for i in range(0,no_lines):
                codes_to_save.append([codes[i]])
                strings_to_save.append([found_codes[i]])
        else:
            if len(codes) < no_lines:
#                print('too few found, expected: '+str(no_lines))
                for i in range(0,no_lines):
                    try:
                        codes_to_save.append([codes[i]])
                    except IndexError:
                        codes_to_save.append([])
                    continue
            else:
#                print('too many found, sort somehow. expected: '+str(no_lines))
                #sorted_list = order_line(training_code_freq_list, codes)
                #print(sorted_list)
                ## if expecting only one code, check for identical matches first
                if no_lines == 1:
                    ## check if identical, then there should only be one code?
                    tmp = []
                    c = 0
                    for cs in found_codes:
                        for s in cs:
                            if s==rr:
                                #print('identical')
                                tmp.append(codes[c])
                        c+=1
                    if len(tmp)>0:
                        sorted_list = order_line(training_code_freq_list, tmp)
                    else:
                        sorted_list = order_line(training_code_freq_list,codes)
                    codes_to_save.append(sorted_list)

                else:
                    ## expecting more than one codes
                    ## find position in raw string from matched code strings
                    ## find longest match
                    ## or sort by chapter?
                    positions = defaultdict()
                    
                    c = 0
                    for cs in found_codes:
                        for s in cs:
                            positions.setdefault(rr.find(s),set())
                            positions[rr.find(s)].add(codes[c])
                        c+=1
                    c = 1
                    for p in sorted(positions):
                        if c <=no_lines:
                            codes_to_save.append(order_line(training_code_freq_list,list(positions[p])))
                        c+=1
                    if len(codes_to_save)<no_lines:
                        for i in range(0,(no_lines-len(codes_to_save))):
                            codes_to_save.append([])
        #print('should be same now, no of lines: '+str(no_lines)+' and codes to add: '+str(len(codes_to_save)))
        ## add new column back to dataframe
        c = 0
        for a in all_info:
            a.append(codes_to_save[c])
            new_lines_for_dataframe.append(a)
            c+=1
        docidalldict[doclineid] = all_info
        #print(docidalldict[doclineid])



        codes_to_add.append(codes_to_save)
#        full_codes_to_add.append(found_codes)
#    return codes_to_add, full_codes_to_add
#    print('New set to save? '+str(len(new_lines_for_dataframe)))
    #new_df = pd.DataFrame(columns=['DocID','YearCoded','Gender','Age', 'LocationOfDeath','LineID','RawText','IntType','IntValue','CauseRank','StandardText','ICD10','dictionary_lookup'])
    new_df = pd.DataFrame(new_lines_for_dataframe)
    if training:
        new_df.columns=['DocID','YearCoded','Gender','Age', 'LocationOfDeath','LineID','RawText','IntType','IntValue','CauseRank','StandardText','ICD10','dictionary_lookup']
    else:
        #DocID;YearCoded;Gender;Age;LocationOfDeath;LineID;RawText;IntType;IntValue
        new_df.columns=['DocID','YearCoded','Gender','Age', 'LocationOfDeath','LineID','RawText','IntType','IntValue','dictionary_lookup']
    new_df['DocID'].astype(int)
    new_df['LineID'].astype(int)
    new_df = new_df.sort_values(['DocID', 'LineID'], ascending=[True, True])
    print(new_df.head())
    return new_df


def find_codes_in_raw_text(rawstrings, icdcodes_strings,stemming=True, ignorediacritics=True, removestopwords=True, lang='french'):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    log = logging.getLogger('find_codes_in_raw_text')
    log.info('searching for codes in text')
    ## list to save all codes and put back in dataframe
    codes_to_add = []
    ## list also with strings associated with a code and found in a rawstring
    full_codes_to_add = []
    count = 0
    t0 = time.time()
    for rr in rawstrings:
#        rr = rr.lower().strip()
#        rr = rr.upper().strip()
        codes = []
        found_codes = []
        if ((count % 1000) == 0):
            t1 = time.time()
            log.info('took: '+(str(datetime.timedelta(seconds=(t1-t0)))))
            log.info(str(count)+' out of: '+str(len(rawstrings)))
            t0 = time.time()
        #log.info(str(count)+' out of: '+str(len(rawstrings)))
        count+=1
        #log.info("rawstring: "+rr)
        try:
            string_p = get_string_ngrams(rr, stemming, ignorediacritics, removestopwords, lang)
        except:
            log.error('Error with line: '+str(count)+' ('+rr+')')
        #print(string_p)
        ## loop over icdcodes
        for i in icdcodes_strings:
            identical = False
            #print(i)
            ## get all strings associated with this code
            codestrings = icdcodes_strings[i]
            ## check if any string is subset of rawstring by getting a set intersect
            intersect = (set(string_p) & set(codestrings))
            ## if the intersect is not an empty set, there was a match
            if len(intersect)>0:
                ## check if the two strings are identical, if so, no need to add a bunch of other matches
                identical = False
                for m in intersect:
                    if rr == m:
                        #log.info('identical, saving: '+m+'('+i+')')
                        found_codes.append([m])
                        codes.append(i)
                        identical=True
                #print('intersect: '+rr+':'+str(list(intersect)))
                #print(i)
                if not identical:
                    found_codes.append(list(intersect))
                    codes.append(i)

        #log.info('found strings in icdcodes: '+str(found_codes))
        #log.info('found codes: '+str(codes))
        codes_to_add.append(codes)
        full_codes_to_add.append(found_codes)
    return codes_to_add, full_codes_to_add


def order_line(dic, icd_codes):
    #create small dic
    d2 = {}
    unseen = []
    for cod in icd_codes:
            if cod in dic:
                d2[dic[cod]]=cod
            else:
                unseen.append(cod)
    #create ordered list
    d2_ordered = sorted(d2)
    sorted_list = []
    for position in d2_ordered:
        sorted_list.append(d2[position])     
    for cod in unseen:   
        sorted_list.append(cod) 
    return sorted_list

def print_with_list_aligned(df, training_code_freq_list, outfile, training=True):
#    DocID;YearCoded;Gender;Age;LocationOfDeath;LineID;RawText;IntType;IntValue;CauseRank;StandardText;ICD10
    new_out = []
    for k,v in df.iterrows():
        clist = v['dictionary_lookup']
        sorted_list = order_line(training_code_freq_list, clist)

        no = []
        no.append(v['DocID'])
        no.append(v['YearCoded'])
        no.append(v['Gender'])
        no.append(v['Age'])
        no.append(v['LocationOfDeath'])
        no.append(v['LineID'])
        no.append(v['RawText'])
        no.append(v['IntType'])
        no.append(v['IntValue'])
        if training:
            no.append(v['CauseRank'])
            no.append(v['StandardText'])
            no.append(v['ICD10'])
        no.append(sorted_list)
#        no.append(v['full_dictionary_lookup'])
        #print("original: "+str(clist))
        #print("sorted: "+str(sorted_list))
        
        new_out.append(no)
#    #print(new_out)
#    tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','Age', 'Gender', 'LocationOfDeath','LineID','RawText','IntType','IntValue','CauseRank','StandardText','ICD10'])
#    tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','Gender','Age', 'LocationOfDeath','LineID','RawText','IntType','IntValue','CauseRank','StandardText','ICD10','dictionary_lookup', 'full_dictionary_lookup'])
    if training:
        tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','Gender','Age', 'LocationOfDeath','LineID','RawText','IntType','IntValue','CauseRank','StandardText','ICD10','dictionary_lookup'])
    else:
        tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','Gender','Age', 'LocationOfDeath','LineID','RawText','IntType','IntValue','dictionary_lookup'])
    tmp2.to_csv(outfile,sep=';', index=False)

def print_line_by_line_for_eval_aligned(df, training_code_freq_list, outfile, training=True):
    new_out = []
    print(len(df))
    for k,v in df.iterrows():
        clist = v['dictionary_lookup']
        sorted_list = order_line(training_code_freq_list, clist)
        if len(sorted_list)>0:
            for c in sorted_list:
                no = []
                no.append(v['DocID'])
                no.append(v['YearCoded'])
                no.append(v['Gender'])
                no.append(v['Age'])
                no.append(v['LocationOfDeath'])
                no.append(v['LineID'])
                no.append(v['RawText'])
                no.append(v['IntType'])
                no.append(v['IntValue'])
                if training:
                    no.append(v['CauseRank'])
                    no.append(v['StandardText'])
                no.append(c)
#            #print(str(c))
                new_out.append(no)
        else:
            no = []
            no.append(v['DocID'])
            no.append(v['YearCoded'])
            no.append(v['Gender'])
            no.append(v['Age'])
            no.append(v['LocationOfDeath'])
            no.append(v['LineID'])
            no.append(v['RawText'])
            no.append(v['IntType'])
            no.append(v['IntValue'])
            if training:
                no.append(v['CauseRank'])
                no.append(v['StandardText'])
            no.append('NULL')
            new_out.append(no)
    print(len(new_out))
#    tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','Age', 'Gender', 'LocationOfDeath','LineID','RawText','IntType','IntValue','CauseRank','StandardText','ICD10'])
    if training:
        tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','Gender','Age', 'LocationOfDeath','LineID','RawText','IntType','IntValue','CauseRank','StandardText','ICD10'])
    else:
        tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','Gender','Age', 'LocationOfDeath','LineID','RawText','IntType','IntValue','ICD10'])

    tmp2.to_csv(outfile,sep=';', index=False)

def print_with_list_raw_fr(df, training_code_freq_list, outfile):
    #DocID;YearCoded;LineID;RawText;IntType;IntValue
    new_out = []
    for k,v in df.iterrows():
        clist = v['dictionary_lookup']
        sorted_list = order_line(training_code_freq_list, clist)

        no = []
        no.append(v['DocID'])
        no.append(v['YearCoded'])
        no.append(v['LineID'])
        no.append(v['RawText'])
        no.append(v['IntType'])
        no.append(v['IntValue'])
        no.append(sorted_list)
        #print("original: "+str(clist))
        #print("sorted: "+str(sorted_list))
        
        new_out.append(no)
#    #print(new_out)
#    tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','Age', 'Gender', 'LocationOfDeath','LineID','RawText','IntType','IntValue','CauseRank','StandardText','ICD10'])
#    tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','Age', 'Gender', 'LocationOfDeath','LineID','RawText','IntType','IntValue','CauseRank','StandardText','dictionary_lookup'])
    tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','LineID','RawText','IntType','IntValue','dictionary_lookup'])

    tmp2.to_csv(outfile,sep=';', index=False)



def print_with_list_raw_it(df, training_code_freq_list, outfile):
    #DocID;YearCoded;LineID;RawText;IntervalText
    new_out = []
    for k,v in df.iterrows():
        clist = v['dictionary_lookup']
        sorted_list = order_line(training_code_freq_list, clist)

        no = []
        no.append(v['DocID'])
        no.append(v['YearCoded'])
        no.append(v['LineID'])
        no.append(v['RawText'])
        no.append(v['IntervalText'])
        no.append(sorted_list)
        #print("original: "+str(clist))
        #print("sorted: "+str(sorted_list))
        
        new_out.append(no)
#    #print(new_out)
#    tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','Age', 'Gender', 'LocationOfDeath','LineID','RawText','IntType','IntValue','CauseRank','StandardText','ICD10'])
#    tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','Age', 'Gender', 'LocationOfDeath','LineID','RawText','IntType','IntValue','CauseRank','StandardText','dictionary_lookup'])
    tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','LineID','RawText','IntervalText','dictionary_lookup'])

    tmp2.to_csv(outfile,sep=';', index=False)

def print_line_by_line_raw_fr(df, training_code_freq_list, outfile):
    new_out = []
    print(len(df))
    for k,v in df.iterrows():
        sorted_list = v['dictionary_lookup']
        #sorted_list = order_line(training_code_freq_list, clist)
        i = 0
        if len(sorted_list)>0:
            for c in sorted_list:
                no = []
                no.append(v['DocID'])
                no.append(v['YearCoded'])
                no.append(v['LineID'])
                no.append(1)
                no.append('NULL')
                no.append(c)
                i+=1
#            #print(str(c))
                new_out.append(no)
        else:
            no = []
            no.append(v['DocID'])
            no.append(v['YearCoded'])
            no.append(v['LineID'])
            no.append(1)
            no.append('NULL')
            no.append('NULL')
            new_out.append(no)
    print(len(new_out))
#    tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','Age', 'Gender', 'LocationOfDeath','LineID','RawText','IntType','IntValue','CauseRank','StandardText','ICD10'])
    tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','LineID','Rank','StandardText','ICD10'])

    tmp2.to_csv(outfile,sep=';', index=False)

def print_line_by_line_raw_it(df, training_code_freq_list, outfile):
    #DocID;YearCoded;LineID;Rank;StandardText;ICD10;IntervalText
    new_out = []
    print(len(df))
    for k,v in df.iterrows():
        sorted_list = v['dictionary_lookup']
        #sorted_list = order_line(training_code_freq_list, clist)
        i = 0
        if len(sorted_list)>0:
            for c in sorted_list:
                no = []
                no.append(v['DocID'])
                no.append(v['YearCoded'])
                no.append(v['LineID'])
                no.append(1)
                no.append('NULL')
                no.append(c)
                no.append('NULL')
                i+=1
#            #print(str(c))
                new_out.append(no)
        else:
            no = []
            no.append(v['DocID'])
            no.append(v['YearCoded'])
            no.append(v['LineID'])
            no.append(1)
            no.append('NULL')
            no.append('NULL')
            no.append('NULL')
            new_out.append(no)
    print(len(new_out))
#    tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','Age', 'Gender', 'LocationOfDeath','LineID','RawText','IntType','IntValue','CauseRank','StandardText','ICD10'])
    tmp2 = pd.DataFrame(new_out, columns=['DocID','YearCoded','LineID','Rank','StandardText','ICD10','IntervalText'])

    tmp2.to_csv(outfile,sep=';', index=False)

def get_dictionary_from_files(file_list, encoding='utf8'):
    df = pd.read_csv(file_list[0], sep=';', error_bad_lines=False, warn_bad_lines=True,encoding=encoding) 
    for f in file_list[1:len(file_list)]:
        tmp = pd.read_csv(f, sep=';', error_bad_lines=False, warn_bad_lines=True,encoding=encoding)
        df = pd.concat([df, tmp]).drop_duplicates().reset_index(drop=True)
    return df


