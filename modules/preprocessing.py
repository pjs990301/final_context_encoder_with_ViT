from scipy import signal
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm
import os

null_subcarrier = [
    '_' + str(x + 32) for x in [-32, -31, -30, -29, -28, -27, 0, 27, 28, 29, 30, 31]]
pilot_subcarrier = ['_' + str(x + 32) for x in [-21, -7, 7, 21]]
additional_subcarrier = ['_' + str(x + 32) for x in [-1, 1, -26, 26]]
unnecessary_columns = ['mac', 'time']

def db(x):
    return 10 * np.log10(x)

def lowpass(csi_vec: np.array, cutoff: float, fs: float, order: int) -> np.array:
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return signal.filtfilt(b, a, csi_vec)

def highpass(csi_vec: np.array, cutoff: float, fs: float, order: int) -> np.array:
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    return signal.filtfilt(b, a, csi_vec)

def bandpass(csi_vec: np.array, low_cut: float, high_cut: float, fs: float, order: int) -> np.array:
    nyq = 0.5*fs
    b, a = signal.butter(order, [low_cut/nyq, high_cut/nyq], btype="band", analog=False)
    return signal.filtfilt(b, a, csi_vec)

def hampel(csi: np.array, k: int=3, nsigma: int=3) -> np.array:
    index = 0
    csi = csi.copy()
    for x in csi:
        y = 0
        if index <= k:
            #Special case, first few samples.
            y = k
        elif index+k > len(csi):
            #Special case, last few samples
            y = -k

        index += y
        stdev = np.std(csi[index-k:index+k])
        median = np.median(csi[index-k:index+k])
        index -= y

        if abs(x-median) > nsigma * stdev:
            csi[index] = median
        index += 1

    return csi

def preprocess(data, args):
    for location in data:
        for label in data[location]:
            for RPI in data[location][label]:
                selected_data = data[location][label][RPI] 
    
                # 실제로 존재하는 컬럼만 필터링
                existing_columns = [col for col in unnecessary_columns if col in selected_data.columns]

                # dBm 변환 적용
                if 'db' in args.preprocess_type:
                    dbm_data = selected_data.drop(columns=existing_columns).applymap(db)
                    selected_data = pd.concat([selected_data[existing_columns], dbm_data], axis=1)

                # 저역 필터링 적용
                if 'lowpass' in args.preprocess_type:
                    low_cut = args.low_cut_off
                    fs = args.low_fs
                    order = args.low_order

                    filtered_data = selected_data.drop(columns=existing_columns).apply(
                        lambda col: lowpass(col.values, low_cut, fs, order), axis=0
                    )
                    selected_data = pd.concat([selected_data[existing_columns], filtered_data], axis=1)

                # 필터링된 데이터를 원래 데이터에 다시 저장
                data[location][label][RPI] = selected_data
                
    return data

def pca(data, args):
    pca = PCA(n_components=args.PCA_components)

    for location in data:
        for label in data[location]:
            if args.is_split:
                selected_data = data[location][label]
                pca_result = pca.fit_transform(selected_data)
                pca_result = pd.DataFrame(pca_result, columns=[f'pca_{i}' for i in range(args.PCA_components)])

                data[location][label] = pca_result

            else:
                for RPI in data[location][label]:
                    selected_data = data[location][label][RPI]
                    pca_result = pca.fit_transform(selected_data.drop(columns=unnecessary_columns))
                    pca_result = pd.DataFrame(pca_result, columns=[f'pca_{i}' for i in range(args.PCA_components)])

                    data[location][label][RPI] = pca_result

    return data
        
def time_mod(data, args, mapping):
    from tqdm import tqdm

    RPI_list = args.RPI

    # for location in data:
    for location in tqdm(data.keys(), desc='Processing Locations'):

        # for label in data[location]:
        for label in tqdm(data[location].keys(), desc=f'Processing Labels in {location}', leave=False):

            # 같은 location과 action에 대해서 RPI1,2,3 데이터 병합
            concatenated_data = pd.DataFrame()
            for RPI in data[location][label]:
                RPI_data = data[location][label][RPI]
                
                if len(RPI_list) == 1:
                    RPI_data['RPI'] = mapping[args.dataset]['encode'].get(str(RPI_list))
                else:
                    RPI_data['RPI'] = mapping[args.dataset]['encode'].get(str(RPI_list)).get(RPI[3])

                concatenated_data = pd.concat([concatenated_data, RPI_data], axis=0)

            # RPI1, RPI2, RPI3이 모두 동일한 시간에 수집을 하였지만 약간의 차이가 발생할 수 밖에 없음
            # 이를 우선 시간에 따른 정렬을 시행
            concatenated_data = concatenated_data.sort_values(by=['time'])
            # os.makedirs(f'{args.output_dir}/scheduling/{args.round_robin_size}/base', exist_ok=True)
            # concatenated_data.reset_index(drop=True).to_csv(f'{args.output_dir}/scheduling/{args.round_robin_size}/base/{location}_{label}_base.csv')
            
            # 시간을 바탕으로 데이터 선택을 진행
            concatenated_data = select_rows_by_rpi_updated(args, concatenated_data)
            # os.makedirs(f'{args.output_dir}/scheduling/{args.round_robin_size}/time', exist_ok=True)
            # concatenated_data.reset_index(drop=True).to_csv(f'{args.output_dir}/scheduling/{args.round_robin_size}/time/{location}_{label}_time.csv')
            
            concatenated_data = concatenated_data.drop(unnecessary_columns + ['RPI'], axis=1).reset_index(drop=True)

            # 선택된 데이터는 location_label이 같음
            data[location][label] = concatenated_data

    return data


def select_rows_by_rpi_updated(args, concatenated_data):
    is_zero_padding = args.is_zero_padding
    round_robin = args.round_robin_size
    is_round_robin_order = args.is_round_robin_order
    round_robin_order = args.round_robin_order    

    selected_rows = pd.DataFrame()

    # 각 라즈베리파이에서 선택한 마지막 시간을 기록
    rpi_present = concatenated_data['RPI'].unique()
    # last_selected_times = {rpi_number: None for rpi_number in range(0, number_RPI)}
    last_selected_times = {rpi_number: None for rpi_number in rpi_present}

    # 만일 라운드로빈 순서가 정해져있다면
    if is_round_robin_order:
        # 정해진 순서대로 진행
        round_robin_order = round_robin_order
    else:
        # 기본적인 라즈베리파이번호 순으로 진행
        round_robin_order = range(0, len(args.RPI))
    
    from math import ceil
    # Estimate the total number of iterations
    num_iterations_per_rpi = {}
    for rpi_number in round_robin_order:
        if rpi_number in rpi_present:
            num_rows = len(concatenated_data[concatenated_data['RPI'] == rpi_number])
        else:
            num_rows = 0  # Zero-padding will add rows for missing RPIs
        num_iterations = ceil(num_rows / round_robin)
        num_iterations_per_rpi[rpi_number] = num_iterations

    # Total iterations is the maximum number of iterations among all RPIs
    total_iterations = max(num_iterations_per_rpi.values())/len(rpi_present)
    
    with tqdm(total=total_iterations, desc='Total Progress', leave=False) as pbar:
        if is_zero_padding:
            while True:
                all_groups_empty = True
                for rpi_number in round_robin_order:
                    # round_robin_order의 rpi_number가 rpi_present에 있는지 확인
                    if rpi_number in rpi_present:
                        # 각 RPI별로 이전에 선택된 'time' 이후의 데이터만 필터링
                        if last_selected_times[rpi_number] is not None:
                            rpi_rows = concatenated_data[(concatenated_data['RPI'] == rpi_number) &
                                                        (concatenated_data['time'] > last_selected_times[rpi_number])]
                        else:
                            rpi_rows = concatenated_data[concatenated_data['RPI'] == rpi_number]

                        if not rpi_rows.empty:
                            all_groups_empty = False
                            selected_rpi_rows = rpi_rows.head(round_robin)
                            selected_rows = pd.concat([selected_rows, selected_rpi_rows], axis=0)

                            # 마지막으로 선택된 'time' 업데이트
                            last_selected_times[rpi_number] = selected_rpi_rows['time'].max()

                            # 이미 선택된 행들 제외
                            concatenated_data = concatenated_data.drop(selected_rpi_rows.index)
                    else:
                        # RPI가 데이터셋에 없으면 제로 패딩 추가
                        last_selected_times[rpi_number] = 1
                        max_time = max([time for time in last_selected_times.values() if time is not None])
                        zero_padding = pd.DataFrame(0, index=np.arange(round_robin), columns=concatenated_data.columns)
                        zero_padding['RPI'] = rpi_number
                        zero_padding['time'] = max_time
                        selected_rows = pd.concat([selected_rows, zero_padding], axis=0)
                        last_selected_times[rpi_number] = zero_padding['time'].max()
                
                # Update the progress bar for each iteration of the while loop
                pbar.update(1)
                
                if all_groups_empty:
                    break
        else:
            while True:
                all_groups_empty = True
                round_robin_order = rpi_present
                for rpi_number in round_robin_order:
                    # 각 RPI별로 이전에 선택된 'time' 이후의 데이터만 필터링
                    if last_selected_times[rpi_number] is not None:
                        rpi_rows = concatenated_data[(concatenated_data['RPI'] == rpi_number) &
                                                    (concatenated_data['time'] > last_selected_times[rpi_number])]
                    else:
                        rpi_rows = concatenated_data[concatenated_data['RPI'] == rpi_number]

                    if not rpi_rows.empty:
                        all_groups_empty = False
                        selected_rpi_rows = rpi_rows.head(round_robin)
                        selected_rows = pd.concat([selected_rows, selected_rpi_rows], axis=0)

                        # 마지막으로 선택된 'time' 업데이트
                        last_selected_times[rpi_number] = selected_rpi_rows['time'].max()

                        # 이미 선택된 행들 제외
                        concatenated_data = concatenated_data.drop(selected_rpi_rows.index)

                # Update the progress bar for each iteration of the while loop
                pbar.update(1)
                
                if all_groups_empty:
                    break

    return selected_rows