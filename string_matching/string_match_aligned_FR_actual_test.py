# coding: utf-8
import sys
import os
import pandas as pd
import time
import datetime
#sys.path.append('/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/')
import string_matching_functions_for_clef_ehealth_icd as smatch


if __name__ == "__main__":    
    #### INPUT DICTIONARY FILES #####
    ### IMPORTANT NOTE: header missing in some dictionaries, have added manually 
    print('creating icd-code dictionary')

    dictfolder = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/dictionaries/'
    dictionaries = []
    dictionaries.append(os.path.join(dictfolder,'Dictionnaire2006-2010.csv'))
    dictionaries.append(os.path.join(dictfolder,'Dictionnaire2011.csv'))
    dictionaries.append(os.path.join(dictfolder,'Dictionnaire2012.csv'))
    dictionaries.append(os.path.join(dictfolder,'Dictionnaire2013.csv'))
    dictionaries.append(os.path.join(dictfolder,'Dictionnaire2014.csv'))
    dictionaries.append(os.path.join(dictfolder,'Dictionnaire2015.csv'))
    df2 = smatch.get_dictionary_from_files(dictionaries)
    ### Generate a dictionary of icd codes and their associated strings from the dictionaries
    icdcodes_strings = smatch.get_code_string_dict(df2, stemming=True, ignorediacritics=True, removestopwords=True, lang='french')

    ## get frequency list from training data
    ## file with training data:
    freqf = '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/training/aligned/corpus/AlignedCauses_2006-2012full.csv'
    training_code_freq_list = smatch.create_dic(freqf)

    ## base folder ##
    folder =  '/Users/sumithra/DSV/MeDESTO/CLEFeHealthSharedTask2018/FR/test/aligned/corpus/'
    #### INPUT FILES #####
    files_to_match = []
    ## aligned  
    files_to_match.append('AlignedCauses_2015F_1')


    file_extension = '.csv'
    for f in files_to_match:
        textf = os.path.join(folder,f+file_extension)
        print('Running on: '+textf)
        df = pd.read_csv(textf, sep=';', error_bad_lines=False, warn_bad_lines=True)  
        ### Get the text from dataframe
        #rawstrings = df['RawText'].tolist()  
    
        t0 = time.time()
        ## find all codes and strings in text ##
        new_df = smatch.find_codes_in_text_aligned(df, icdcodes_strings, freqf, stemming=True, ignorediacritics=True, removestopwords=True, lang='french', training=False)
        

        outfolder = os.path.join(folder,'string_matched_full')
        if not os.path.isdir(outfolder):
            os.mkdir(outfolder)

        outf_s = os.path.join(outfolder,f+'_system_list'+file_extension)
        outf_s2 = os.path.join(outfolder,f+'_system'+file_extension)
        smatch.print_with_list_aligned(new_df, training_code_freq_list, outf_s, training=False)
        smatch.print_line_by_line_for_eval_aligned(new_df, training_code_freq_list, outf_s2, training=False)

    
        t1 = time.time()
        print('took: '+(str(datetime.timedelta(seconds=(t1-t0)))))

