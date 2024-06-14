import pandas as pd
import numpy as np
import polars as pl
from polars import selectors as cs
from utils import *
import rcs_cataloging as rcc # --> probably need to build first


def clean_packets(df):
    # Identifying packets for rejection
    medianTimestamp = df.select(pl.col('timestamp')).median().item()
    numSecs = 24*60*60
    
    df = df.filter(
        # Remove packets with timestamp that are more than 24 hours from median timestamp
        (abs(pl.col('timestamp') - medianTimestamp) < numSecs),
        # Remove packets with negative PacketGenTime
        (pl.col('PacketGenTime') > 0),
        # Consecutive packets with identical dataTypeSequence and systemTick;
        # identify the second packet for removal; identify the first
        # instance of these duplicates below
        (pl.col('dataTypeSequence').shift(1) != pl.col('dataTypeSequence'))
    )
    
    # Remove packets with greater than 5 sec difference from elapsed packetgentime and elapsed timestamp
    # Get first PacketGenTime and timestamp
    firstGen = df.select(pl.col('PacketGenTime').first()).item()
    firstTime = df.select(pl.col('timestamp').first()).item()
    
    df = df.filter(
        abs( (pl.col('PacketGenTime') - firstGen)/1000 - (pl.col('timestamp') - firstTime) ) < 5 # divide by 1000 to convert to seconds
    )
    
    # Identify packetGenTimes that go backwards in time by more than 500ms; may overlap with negative PacketGenTime
    # Look up to 6 packets ahead. Remove said packets
    df = df.filter(
        (pl.col('PacketGenTime') - pl.col('PacketGenTime').shift(1) > -500),
        (pl.col('PacketGenTime') - pl.col('PacketGenTime').shift(2) > -500),
        (pl.col('PacketGenTime') - pl.col('PacketGenTime').shift(3) > 0),
        (pl.col('PacketGenTime') - pl.col('PacketGenTime').shift(4) > 0),
        (pl.col('PacketGenTime') - pl.col('PacketGenTime').shift(5) > 0),
        (pl.col('PacketGenTime') - pl.col('PacketGenTime').shift(6) > 0),
    )
    
    return df



def assignTime(df):
    '''
    Assign time to each packet based on systemTick and sample rate. This is a translation of assignTime.m into python.
    '''
    # Might be able to just pass in non-exploded table
    df_packets = df.with_row_index().filter(pl.col('timestamp').is_not_null())
    
    df_packets = clean_packets(df_packets)
    
    # Identify new chunks, assign to new variable
    df_packets = df_packets.with_columns(
        pl.when(
        # Change in sample rate 
        (pl.col('samplerate').shift(1) != pl.col('samplerate') ) | 
        # Timestamps jump by more than 1
        (pl.col('timestamp') - pl.col('timestamp').shift(1) > 1) | 
        # dataTypeSequence doesn't iterate by 1
        (pl.col('dataTypeSequence') - pl.col('dataTypeSequence').shift(1)).is_in([1, -255]).not_()
        ).then(1)
        .otherwise(0).alias('newChunk')
    )
    
    # Identify gaps in packets
    df_packets = df_packets.with_columns(
        ((pl.col('systemTick') + 2**16 - pl.col('systemTick').shift(1)) % 2**16).alias('systemTickDiff'),
        (pl.col('packetsizes') * (1 / pl.col('samplerate')) * 1e4).alias('expectedElapsed') # in milliseconds
    )
    
    # Identify and label new chunks of data
    df_packets = df_packets.with_columns(
        # If diff_systemTick and expectedElapsed differ by more than the larger of 50% of expectedElapsed OR 100ms,
        # flag as gap
        pl.when(
            (abs(pl.col('expectedElapsed') - pl.col('systemTickDiff')) > 1000) |
            (abs(pl.col('expectedElapsed') - pl.col('systemTickDiff')) > (0.5 * pl.col('expectedElapsed')))
        ).then(1).otherwise(pl.col('newChunk')).alias('newChunk')
    ).with_columns(
        pl.col('newChunk').cum_sum().alias('chunkNum')
    )
    
    # left off at line 153 in assignTime.m
    # We will assume a shortGaps_systemTick of 0 for now
    df_packets = df_packets.with_columns(
        (pl.col('PacketGenTime').diff() * 1e1).alias('diff_PacketGenTime'), # multiply by 1e1 to convert to 1e-4 seconds
    )
    
    # Calculate error before chunking
    df_packets = df_packets.with_columns(
        (pl.col('expectedElapsed') - pl.col('diff_PacketGenTime')).alias('error')
    )
    
    
    df_chunked = df_packets.group_by('chunkNum').agg(
        pl.all().exclude('newChunk', 'chunkNum'),
    )
    
    # Chunks must have at least 2 packets in order to have a valid
    # diff_PacketGenTime -- thus if chunk only one packet, it must be
    # identified. These chunks can remain if the timeGap before is < 6
    # seconds, but must be excluded if the timeGap before is >= 6 seconds
    # or error set to zero
    
    df_chunked = df_chunked.with_columns(
        pl.when(
            pl.col('packetsizes').list.len() == 1
        ).then(pl.lit(0)).otherwise(
            pl.col('error').list.slice(1,None).list.median()
        ).alias('medianError')
    )
    
    # Add corrected time for each chunk
    df_packets = df_chunked.explode(pl.exclude('medianError', 'chunkNum')).with_columns(
        (pl.col('PacketGenTime') + pl.col('medianError')*1e-1).alias('correctedAlignTime')
    )
    
    deltaTime = 1 / df_packets.select(pl.col('samplerate')).max().item() * 1000
    
    df_packets = df_packets.with_columns(
        (
        (((pl.col('correctedAlignTime') - pl.col('correctedAlignTime').first())/deltaTime) + 0.5).floor() * deltaTime + pl.col('correctedAlignTime').first()
        ).alias('correctedAlignTime_shifted')
    )
    
    packets_to_keep = df_packets.get_column('PacketGenTime')
    
    df_out = df.with_row_index().with_columns(
        pl.col('PacketGenTime').backward_fill()
    ).filter(
        pl.col('PacketGenTime').is_in(packets_to_keep)
    )
    
    
    first_align_time = df_packets.select(pl.col('correctedAlignTime_shifted').min()).item()
    first_packet_sr = df_out.select(pl.col('samplerate').first()).item()
    first_packet_ind = df_packets.select(pl.col('index').min()).item()
    
    df_out = (
        df_out
        .join(df_packets.select(['index', 'correctedAlignTime_shifted', 'chunkNum']), on='index', how='left')
        .with_columns(
            pl.col('chunkNum').forward_fill().backward_fill(),
        ).with_columns(
            pl.when(
                pl.col('index') == pl.col('index').first()
            ).then(
                first_align_time - (first_packet_ind - pl.col('index').first()) * (1000/first_packet_sr)
            ).otherwise(pl.col('correctedAlignTime_shifted'))
            .alias('correctedAlignTime_shifted')
        )
    ).with_columns(
        pl.col('correctedAlignTime_shifted').interpolate('linear').alias('DerivedTime')
    )
    
    
    # Needs QC : I used interpolate instead of calculating the Derived time for each chunk.. 
    # may need to better translate matlab code block 378 - 409 (which has beeen started below)
    
    # # Create Derived Time
    # derived_chunks = df_out.partition_by('chunkNum')
    # for chunk in derived_chunks:
    #     sr = chunk.select(pl.col('samplerate').unique()).item()
    #     chunk_tmp = chunk.filter(pl.col('correctedAlignTime_shifted').is_not_null())
    #     elapsedTime_before = (chunk_tmp.select(pl.col('index').first()).item() - chunk.select(pl.col('index').first()).item()) * (1000/sr)
    #     elapsedTime_after = (chunk.select(pl.col('index').last()).item() - chunk_tmp.select(pl.col('index').first()).item()) * (1000/sr)
        
    
    # df_derived = df_out.select()


    return df_out


def process_device_settings(settings):
    """
    Process the device settings json file and return a dictionary with the important keys
    """
    dfs = []
    for dict in settings:
        if (len(dict.keys()) == 2) and ('RecordInfo' in list(dict.keys())) and ('TelemetryModuleInfo' in list(dict.keys())): continue # --> skip these entries, they have no useful information
        if (len(dict.keys()) == 2) and ('RecordInfo' in list(dict.keys())) and ('BatteryStatus' in list(dict.keys())): continue # --> skip these entries, they have no useful information
        if (len(dict.keys()) == 2) and ('RecordInfo' in list(dict.keys())) and ('SubjectInfo' in list(dict.keys())): continue # --> skip these entries, they have no useful information
        entry = {}
        for key, value in dict.items():
            if key == 'RecordInfo':
                entry['HostUnixTime'] = value['HostUnixTime']
                entry['SessionId'] = value['SessionId']
                
            elif key == 'GeneralData':
                entry['ActiveGroup'] = value['therapyStatusData']['activeGroup']
                entry['TherapyStatus'] = value['therapyStatusData']['therapyStatus']
                
            elif key == 'SensingConfig':
                for i, d in enumerate(value['timeDomainChannels']):
                    entry[f'channel_{i}_sensing'] = {k: v for k, v in d.items() if k in ["hpf", "lpf1", "lpf2", "minusInput", "plusInput", "sampleRate"]}
                    
            elif key == 'DetectionConfig':
                ld0 = value['Ld0']
                ld0 = {k: [v] if isinstance(v, list) else v for k, v in ld0.items()}
                entry['Ld0'] = ld0
                
                ld1 = value['Ld1']
                ld1 = {k: [v] if isinstance(v, list) else v for k, v in ld1.items()}
                entry['Ld1'] = ld1
                
            elif 'TherapyConfigGroup' in key:
                if key == f'TherapyConfigGroup{entry.get("ActiveGroup")}':
                    entry['RateInHz'] = value['rateInHz']
                    entry['rampTime'] = value['rampTime']
                    for program in value['programs']:
                        if program['isEnabled'] == 0 and program["valid"]:
                            entry['pulseWidthInMicroseconds'] = program['pulseWidthInMicroseconds']
                            entry['AmplitudeInMilliamps'] = program['amplitudeInMilliamps']
                            for i, dict in enumerate(program['electrodes']['electrodes']):
                                if ~dict['isOff']:
                                    entry['stim_electrode'] = i
                            entry['active_recharge'] = program['miscSettings']['activeRechargeRatio']
            
            elif key == 'AdaptiveConfig':
                entry['adaptiveMode'] = value['adaptiveMode']
                entry['adaptiveStatus'] = value['adaptiveStatus']
                entry['delta_fall'] = value['deltas'][0]['fall']
                entry['delta_fall'] = value['deltas'][0]['fall']
                for sub_key in value.keys():
                    if 'state' in sub_key:
                        sub_dict = {'prog0AmpInMilliamps': value[sub_key]['prog0AmpInMilliamps'], 'rateTargetInHz': value[sub_key]['rateTargetInHz'], 'isValid': value[sub_key]['isValid']}
                        entry[f"Adaptive_{sub_key}"] = sub_dict
            
        if entry:
            entry = {k: [v] for k, v in entry.items()}
            dfs.append(pl.from_dict(entry))
        
    settings = pl.concat(dfs, how='diagonal')
    
    # Remove rows with no relevant information
    cols = settings.columns[2:]
    settings_col = cs.by_name(*cols)
    settings.filter(~pl.all_horizontal(settings_col.is_null())).head()
    
    settings = settings.with_columns(pl.col('SessionId').cast(pl.Categorical))
    return settings


def parse_event_log(event_log):
    """
    Parse the event log json file and return polars dataframe with relevant log info. NOTE: IGNORES BATTERY LOG INFO
    """
    rows = {'SessionId': [], 'HostUnixTime': [], 'EventType': [], 'EvenSubType': []}
    for dict in event_log:
        event = dict['Event']
        if ('LeadLocation' in event['EventType']) or ('Battery' in event['EventType']) or ('Application Version' in event['EventType']):
            continue
        else:
            rows['SessionId'].append(dict['RecordInfo']['SessionId'])
            rows['HostUnixTime'].append(dict['RecordInfo']['HostUnixTime'])
            rows['EventType'].append(event['EventType'])
            rows['EvenSubType'].append(event['EventSubType'])
    
    return pl.DataFrame(rows).with_columns(pl.col('SessionId').cast(pl.Categorical))


def parse_adaptive_log(adaptive_log):
    """
    Parse the adaptive log json file and return polars dataframe with relevant log info
    """
    # Check if adaptive log is empty
    if not adaptive_log:
        return None
    
    rows = {'SessionId': [], 'HostUnixTime': [], 'PacketGenTime': [], 'PacketRxUnixTime': [], 'AdaptiveState': [], 'StimRateInHz': [], 'AmplitudeInMilliamps': [], 'Ld0DetectionStatues': [], 'Ld1DetectionStatues': []}
    for dict in adaptive_log:
        adapt = dict['AdaptiveUpdate']
        rows['SessionId'].append(dict['RecordInfo']['SessionId'])
        rows['HostUnixTime'].append(dict['RecordInfo']['HostUnixTime'])
        rows['PacketGenTime'].append(adapt['PacketGenTime'])
        rows['PacketRxUnixTime'].append(adapt['PacketRxUnixTime'])
        rows['AdaptiveState'].append(adapt['CurrentAdaptiveState'])
        rows['StimRateInHz'].append(adapt['StimRateInHz'])
        rows['AmplitudeInMilliamps'].append(adapt['CurrentProgramAmplitudesInMilliamps'][0])
        rows['Ld0DetectionStatues'].append(adapt['Ld0DetectionStatus'])
        rows['Ld1DetectionStatues'].append(adapt['Ld1DetectionStatus'])
    
    return pl.DataFrame(rows).with_columns(pl.col('SessionId').cast(pl.Categorical))