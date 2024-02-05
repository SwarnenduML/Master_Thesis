import sys

print(sys.argv)
# Check the number of arguments
# if len(sys.argv) != 2:
#     print("Usage: python script.py argument")
#     sys.exit(1)

filename = sys.argv[1]
#ds_type = sys.argv[2]
print(filename)
#print(ds_type)

# all imports come here
from faster_whisper import WhisperModel
import pandas as pd
import torch
import torchaudio
from statistics import mean, median, mode
import openpyxl
import warnings
import numpy as np
import time
import torchcrepe
from scipy.io import wavfile
import pesto
import crepe
import resampy
import logging

logging.basicConfig(filename="/speech/dbwork/mul/spielwiese4/students/desengus/dry_crepe_pesto/tency_dryvocals/logging.log", filemode='w', format='%(name)s %(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('Some training')
logger.setLevel(logging.DEBUG)

crepe_option = True
pesto_option = True

start = time.time()
warnings.filterwarnings("ignore")

# all major variables are here
# get the current time in seconds since the epoch

# all major variables are here
model_size = "medium.en"
music_file_path = '/speech/dbwork/mul/spielwiese3/students/debaumas/datasets/tency_mastering_supervised_dry_wet_v2_22050_16bit/train/dry/0.wav'
sample_rate_speech = 16000
step_size_for_pitch = 10.
vad_threshold = 0.3 # speech probability - if more then speech else no speech
confidence_pesto = 0.1
num_sample = 10 # pesto computes 10 ms per sample so for a window of 100ms we need 10 samples as 10*10ms = 100ms
number_per_sample = 200 #200*100ms = 20sec sampling
timestamp_for_sampling = step_size_for_pitch*num_sample/1000
sec_sampling = 2 #time for each sample or input - each sample is 50 ms


# generate the modified start and the end times of the words said
# generate the modified start and the end times of the words said
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

# wav = read_audio(music_file_path, sampling_rate=sample_rate_speech)
# speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sample_rate_speech, threshold=vad_threshold)


# Run on GPU with int8
faster_whisper_model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

print("Till HERE DONE")

# music file path is to be given here
with open(filename, 'r') as file:
    # Read the entire content of the file
    for line in file:
        line = line.strip()
        logger.info(line)
        if line[-4:]=='.wav':
            music_file_path = line
            logger.info(music_file_path)
            wav = read_audio(music_file_path, sampling_rate=sample_rate_speech)
            speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sample_rate_speech, threshold=vad_threshold)

            total_datapoints = round(list(wav.shape)[0]/sample_rate_speech,0)

            # get the current time in seconds since the epoch
            seconds = time.time()

            # dataset creation
            tmp_x = len(speech_timestamps)
            word_timings = pd.DataFrame(columns = ['start','end','words'])
            start = []
            end = []
            words = []

            for i in range(tmp_x):
                save_audio('/speech/dbwork/mul/spielwiese4/students/desengus/tmp_tency/'+music_file_path.split("/")[-1][:-4]+'.wav',
                        wav[speech_timestamps[i]['start']:speech_timestamps[i]['end']], sampling_rate=sample_rate_speech)
                segments, _ = faster_whisper_model.transcribe('/speech/dbwork/mul/spielwiese4/students/desengus/tmp_crepe/'+music_file_path.split("/")[-1][:-4]+'.wav', word_timestamps=True)
                for segment in segments:
                    for word in segment.words:
                        start.append(round(word.start+speech_timestamps[i]['start']/sample_rate_speech,3))
                        end.append(round(word.end+speech_timestamps[i]['end']/sample_rate_speech, 3))
                        words.append(word.word)

                #print(round(word.start+speech_timestamps[i]['start']/sample_rate_speech,4), round(word.end+speech_timestamps[i]['end']/sample_rate_speech, 4), word.word)

            word_timings['start'] = start
            word_timings['end'] = end
            word_timings['words'] = words
            print(word_timings.shape)

            if word_timings.shape[0]>0:
                if pesto_option:
                    # predict the pitch of your audio tensors directly within your own Python code PESTO
                    audio, sample_rate_pesto = torchaudio.load(music_file_path)
                    audio = torchaudio.functional.resample(audio, orig_freq=sample_rate_pesto, new_freq=sample_rate_speech)
                    timesteps_pesto, pitch_pesto, confidence_pesto, activations_pesto = pesto.predict(audio, sample_rate_speech, step_size=step_size_for_pitch)

                    timesteps_result_pesto = [round(tensor.item(),2) for tensor in timesteps_pesto]
                    pitch_result_pesto = [round(tensor.item(),4) for tensor in pitch_pesto]
                    confidence_result_pesto = [round(tensor.item(),4) for tensor in confidence_pesto]
                    df_pitch_pesto = pd.DataFrame([timesteps_result_pesto,pitch_result_pesto,confidence_result_pesto])
                    df_pitch_pesto = df_pitch_pesto.T
                    df_pitch_pesto.columns = ['timesteps_pesto','pitch_pesto','confidence_pesto']


                if crepe_option:
                    # predict the pitch of your audio tensors directly within your own Python code - CREPE
                    sample_rate, audio = wavfile.read(music_file_path)
                    audio = resampy.resample(audio, sr_orig=sample_rate, sr_new=sample_rate_speech)
                    timesteps_crepe, pitch_crepe, confidence_crepe, activations_crepe = crepe.predict(audio, sample_rate_speech, step_size=step_size_for_pitch)

                    timesteps_result_crepe = [round(tensor.item(),2) for tensor in timesteps_crepe]
                    pitch_result_crepe = [round(tensor.item(),4) for tensor in pitch_crepe]
                    confidence_result_crepe = [round(tensor.item(),4) for tensor in confidence_crepe]
                    df_pitch_crepe = pd.DataFrame([timesteps_result_crepe,pitch_result_crepe,confidence_result_crepe])
                    df_pitch_crepe = df_pitch_crepe.T
                    df_pitch_crepe.columns = ['timesteps_crepe','pitch_crepe','confidence_crepe']


                # Iterate through the rows and update 'end' based on the condition
                # it is done such that the end of 1st tone is before or at the start of the second one
                # set the end based on the start of the next word such that there is no overlap
                for i in range(len(word_timings) - 1):
                    if word_timings.at[i, 'end'] > word_timings.at[i + 1, 'start']:
                        word_timings.at[i, 'end'] = word_timings.at[i + 1, 'start']


                # creating more features
                word_timings['start'] = round(word_timings['start'],1)
                word_timings['end'] = round(word_timings['end'],1)
                if pesto_option:
                    word_timings['pitches_pesto'] = pd.Series(dtype='object')
                    word_timings['mean_pitches_pesto'] = pd.Series(dtype='object')
                    word_timings['median_pitches_pesto'] = pd.Series(dtype='object')
                    word_timings['mode_pitches_pesto'] = pd.Series(dtype='object')
                if crepe_option:
                    word_timings['pitches_crepe'] = pd.Series(dtype='object')
                    word_timings['mean_pitches_crepe'] = pd.Series(dtype='object')
                    word_timings['median_pitches_crepe'] = pd.Series(dtype='object')
                    word_timings['mode_pitches_crepe'] = pd.Series(dtype='object')
                word_timings['samples'] = (word_timings['end']-word_timings['start'])/timestamp_for_sampling
                word_timings['samples'] = word_timings['samples'].round(0).astype('int')
                word_timings['act_start'] = pd.Series(dtype='object')
                word_timings['act_end'] = pd.Series(dtype='object')
                word_timings['word_index'] = word_timings.index+1
                word_timings['word_index'] = word_timings['word_index'].astype(str)

                if pesto_option:
                    word_timings['mean_note_pesto'] = pd.Series(dtype='object')
                    word_timings['median_note_pesto'] = pd.Series(dtype='object')
                    word_timings['mode_note_pesto'] = pd.Series(dtype='object')
                if crepe_option:
                    word_timings['mean_note_crepe'] = pd.Series(dtype='object')
                    word_timings['median_note_crepe'] = pd.Series(dtype='object')
                    word_timings['mode_note_crepe'] = pd.Series(dtype='object')

                print("word_timings made")
                # Duplicate rows based on the 'Value' column
                word_timings_expanded = word_timings.loc[word_timings.index.repeat(word_timings['samples'])]

                # Reset index to get a new DataFrame with duplicated rows
                word_timings_expanded = word_timings_expanded.reset_index(drop=True)

                # creating the sample timestamps
                tmp_start = word_timings_expanded['start'][0]
                word_timings_expanded['act_start'][0] = tmp_start
                word_timings_expanded['act_end'][0] = tmp_start + timestamp_for_sampling
                for i in range(1,word_timings_expanded.shape[0]):
                    if tmp_start == word_timings_expanded['start'][i]:
                        word_timings_expanded['act_start'][i] = (word_timings_expanded['act_end'][i-1]).round(1)
                        word_timings_expanded['act_end'][i] = (word_timings_expanded['act_start'][i]+timestamp_for_sampling).round(1)
                    else:
                        tmp_start = word_timings_expanded['start'][i]
                        word_timings_expanded['act_start'][i] = (tmp_start).round(1)
                        word_timings_expanded['act_end'][i] = (word_timings_expanded['act_start'][i]+timestamp_for_sampling).round(1)
        
                print("Sample timings made")

                # setting the pitches for PESTO and CREPE
                for i in range(word_timings_expanded.shape[0]):
                    start = word_timings_expanded['act_start'][i]
                    end = word_timings_expanded['act_end'][i]
                    pitch_pesto = []
                    pitch_crepe = []
                    #print(i)
                    if pesto_option:
                        for j in range(df_pitch_pesto.shape[0]):
                            if start <= df_pitch_pesto['timesteps_pesto'][j] <= end:
                                pitch_pesto.append(df_pitch_pesto['pitch_pesto'][j])
                        word_timings_expanded['pitches_pesto'][i] = pitch_pesto

                    if crepe_option:
                        for j in range(df_pitch_crepe.shape[0]):
                            if start <= df_pitch_crepe['timesteps_crepe'][j] <= end:
                                pitch_crepe.append(df_pitch_crepe['pitch_crepe'][j])
                        word_timings_expanded['pitches_crepe'][i] = pitch_crepe

                print("pitches done")

                # statistical acquiring of the pitches
                for i in range(word_timings_expanded.shape[0]):
                    #print(i)
                    if pesto_option:
                        if word_timings_expanded['pitches_pesto'][i] ==[]:
                            word_timings_expanded['mean_pitches_pesto'][i] = 0
                            word_timings_expanded['median_pitches_pesto'][i] = 0
                            word_timings_expanded['mode_pitches_pesto'][i] = 0
                        else:
                            word_timings_expanded['mean_pitches_pesto'][i] = mean(word_timings_expanded['pitches_pesto'][i])
                            word_timings_expanded['median_pitches_pesto'][i] = median(word_timings_expanded['pitches_pesto'][i])
                            word_timings_expanded['mode_pitches_pesto'][i] = mode(word_timings_expanded['pitches_pesto'][i])

                    if crepe_option:
                        if word_timings_expanded['pitches_crepe'][i] ==[]:
                            word_timings_expanded['mean_pitches_crepe'][i] = 0
                            word_timings_expanded['median_pitches_crepe'][i] = 0
                            word_timings_expanded['mode_pitches_crepe'][i] = 0
                        else:
                            word_timings_expanded['mean_pitches_crepe'][i] = mean(word_timings_expanded['pitches_crepe'][i])
                            word_timings_expanded['median_pitches_crepe'][i] = median(word_timings_expanded['pitches_crepe'][i])
                            word_timings_expanded['mode_pitches_crepe'][i] = mode(word_timings_expanded['pitches_crepe'][i])

                print("stat done")


                # note to frequency conversion
                note_ds = pd.DataFrame()
                note_ds['Note'] = ['0','C0','C0#','D0','D0#','E0','F0','F0#','G0','G0#','A0','A0#','B0',
                                'C1','C1#','D1','D1#','E1','F1','F1#','G1','G1#','A1','A1#','B1',
                                'C2','C2#','D2','D2#','E2','F2','F2#','G2','G2#','A2','A2#','B2',
                                'C3','C3#','D3','D3#','E3','F3','F3#','G3','G3#','A3','A3#','B3',
                                'C4','C4#','D4','D4#','E4','F4','F4#','G4','G4#','A4','A4#','B4',
                                'C5','C5#','D5','D5#','E5','F5','F5#','G5','G5#','A5','A5#','B5',
                                'C6','C6#','D6','D6#','E6','F6','F6#','G6','G6#','A6','A6#','B6',
                                'C7','C7#','D7','D7#','E7','F7','F7#','G7','G7#','A7','A7#','B7',
                                'C8','C8#','D8','D8#','E8','F8','F8#','G8','G8#','A8','A8#','B8']
                note_ds['Frequency'] = [0, 16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50,
                                            25.96, 27.50, 29.14, 30.87, 32.70, 34.65, 36.71, 38.89,
                                            41.20, 43.65, 46.25, 49.00, 51.91, 55.00, 58.27, 61.74,
                                            65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00,
                                            103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83,
                                            155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00,
                                            233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63,
                                            349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88,
                                            523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99,
                                            783.99, 830.61, 880.00, 932.33, 987.77, 1046.50, 1108.73,
                                            1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98,
                                            1661.22, 1760.00, 1864.66, 1975.53, 2093.00, 2217.46, 2349.32,
                                            2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520.00,
                                            3729.31, 3951.07, 4186.01, 4434.92, 4698.63, 4978.03, 5274.04,
                                            5587.65, 5919.91, 6271.93, 6644.88, 7040.00, 7458.62, 7902.13]

                if crepe_option:
                    word_timings_expanded['mean_note_crepe'] = pd.Series(dtype = 'object')
                    word_timings_expanded['median_note_crepe'] = pd.Series(dtype = 'object')
                    word_timings_expanded['mode_note_crepe'] = pd.Series(dtype = 'object')

                if pesto_option:
                    word_timings_expanded['mean_note_pesto'] = pd.Series(dtype = 'object')
                    word_timings_expanded['median_note_pesto'] = pd.Series(dtype = 'object')
                    word_timings_expanded['mode_note_pesto'] = pd.Series(dtype = 'object')

                for i in range(word_timings_expanded.shape[0]):
                    if pesto_option:
                        mean_s = min(range(len(note_ds['Frequency'])), key=lambda j: abs(note_ds['Frequency'][j]-word_timings_expanded['mean_pitches_pesto'][i]))
                        word_timings_expanded['mean_note_pesto'][i] = note_ds['Note'][mean_s]
                        median_s = min(range(len(note_ds['Frequency'])), key=lambda k: abs(note_ds['Frequency'][k]-word_timings_expanded['median_pitches_pesto'][i]))
                        word_timings_expanded['median_note_pesto'][i] = note_ds['Note'][median_s]
                        mode_s = min(range(len(note_ds['Frequency'])), key=lambda l: abs(note_ds['Frequency'][l]-word_timings_expanded['mode_pitches_pesto'][i]))
                        word_timings_expanded['mode_note_pesto'][i] = note_ds['Note'][mode_s]
                    
                    if crepe_option:
                        mean_s = min(range(len(note_ds['Frequency'])), key=lambda j: abs(note_ds['Frequency'][j]-word_timings_expanded['mean_pitches_crepe'][i]))
                        word_timings_expanded['mean_note_crepe'][i] = note_ds['Note'][mean_s]
                        median_s = min(range(len(note_ds['Frequency'])), key=lambda k: abs(note_ds['Frequency'][k]-word_timings_expanded['median_pitches_crepe'][i]))
                        word_timings_expanded['median_note_crepe'][i] = note_ds['Note'][median_s]
                        mode_s = min(range(len(note_ds['Frequency'])), key=lambda l: abs(note_ds['Frequency'][l]-word_timings_expanded['mode_pitches_crepe'][i]))
                        word_timings_expanded['mode_note_crepe'][i] = note_ds['Note'][mode_s]               
                print("freq conv done")

                # dynamically calculating song length
                song_length = round(list(audio.shape)[0]/sample_rate_speech,2) # song length in sec

                # creating the df for merging
                start_values = np.arange(0, song_length, timestamp_for_sampling)
                end_values = start_values + 0.1

                df = pd.DataFrame({'act_start': start_values, 'act_end': end_values})

                print("creating the df for merging")

                # merging
                word_timings_expanded['act_end'] = word_timings_expanded['act_end'].astype(float).round(1)
                word_timings_expanded['act_start'] = word_timings_expanded['act_start'].astype(float).round(1)
                df['act_end'] = df['act_end'].astype(float).round(1)
                df['act_start'] = df['act_start'].astype(float).round(1)

                merged_df = pd.merge(word_timings_expanded, df, on =['act_start', 'act_end'], how = 'right')
                print("Merging successful")

                col_to_drop = ['start','end','samples']
                if crepe_option:
                    col_to_drop.extend(['pitches_crepe','mean_pitches_crepe','median_pitches_crepe','mode_pitches_crepe'])
                if pesto_option:
                    col_to_drop.extend(['pitches_pesto','mean_pitches_pesto','median_pitches_pesto','mode_pitches_pesto'])

                df = merged_df.drop(columns=col_to_drop)    

                # filling with 0s and Silences

                df['words'] = df['words'].fillna('[silence]')
                df['word_index'] = df['word_index'].fillna('0')
                if pesto_option:
                    df['mean_note_pesto'] = df['mean_note_pesto'].fillna('0')
                    df['median_note_pesto'] = df['median_note_pesto'].fillna('0')
                    df['mode_note_pesto'] = df['mode_note_pesto'].fillna('0')

                if crepe_option:
                    df['mean_note_crepe'] = df['mean_note_crepe'].fillna('0')
                    df['median_note_crepe'] = df['median_note_crepe'].fillna('0')
                    df['mode_note_crepe'] = df['mode_note_crepe'].fillna('0')

                #print(df)
                if df.shape[0]>number_per_sample:
                    # Grouping based on intervals of 2 indexes
                    # Concatenate other columns within each group
                    # Grouping based on intervals of 200 indexes
                    # Concatenate other columns within each group
                    if pesto_option and crepe_option:
                        result_df = df.groupby(df.index // number_per_sample).agg({
                        'words': lambda x: ' ; '.join(pd.unique(x).astype(str)),
                        'word_index': lambda x: ' ; '.join(x.astype(str)),
                        'mean_note_pesto': lambda x: ' ; '.join(x.astype(str)),
                        'median_note_pesto': lambda x: ' ; '.join(x.astype(str)),
                        'mode_note_pesto': lambda x: ' ; '.join(x.astype(str)),
                        'mean_note_crepe': lambda x: ' ; '.join(x.astype(str)),
                        'median_note_crepe': lambda x: ' ; '.join(x.astype(str)),
                        'mode_note_crepe': lambda x: ' ; '.join(x.astype(str))
                    }).reset_index(drop=True)
                    elif pesto_option:
                                                result_df = df.groupby(df.index // number_per_sample).agg({
                        'words': lambda x: ' ; '.join(pd.unique(x).astype(str)),
                        'word_index': lambda x: ' ; '.join(x.astype(str)),
                        'mean_note_pesto': lambda x: ' ; '.join(x.astype(str)),
                        'median_note_pesto': lambda x: ' ; '.join(x.astype(str)),
                        'mode_note_pesto': lambda x: ' ; '.join(x.astype(str))
                    }).reset_index(drop=True)
                    elif crepe_option:
                        result_df = df.groupby(df.index // number_per_sample).agg({
                        'words': lambda x: ' ; '.join(pd.unique(x).astype(str)),
                        'word_index': lambda x: ' ; '.join(x.astype(str)),
                        'mean_note_crepe': lambda x: ' ; '.join(x.astype(str)),
                        'median_note_crepe': lambda x: ' ; '.join(x.astype(str)),
                        'mode_note_crepe': lambda x: ' ; '.join(x.astype(str))
                    }).reset_index(drop=True)
                else:
                    result_df = df

                print("Aggregation done")
                if pesto_option and crepe_option:
                    result_df.to_excel('/speech/dbwork/mul/spielwiese4/students/desengus/dry_crepe_pesto/tency_dryvocals/excels/'+music_file_path.split("/")[-1][:-4]+'.xlsx')
                elif crepe_option:
                    result_df.to_excel('/speech/dbwork/mul/spielwiese4/students/desengus/dry_crepe/tency_dryvocals/excels/'+music_file_path.split("/")[-1][:-4]+'.xlsx')
                elif pesto_option:
                    result_df.to_excel('/speech/dbwork/mul/spielwiese4/students/desengus/dry_pesto/tency_dryvocals/excels/'+music_file_path.split("/")[-1][:-4]+'.xlsx')


                print("line ended")
                print(time.time()-seconds)
