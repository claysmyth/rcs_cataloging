import pandas as pd
import numpy as np
import pandas as pd

def assign_time(input_data_table, short_gaps_system_tick=0):
    """
    Function for creating timestamps for each sample of valid RC+S data. Given
    known limitations of all recorded timestamps, need to use multiple variables
    to derive time.

    General approach: Remove packets with faulty meta-data.
    Identify gaps in data (by checking deltas in timestamp, systemTick,
    and dataTypeSequence). Consecutive packets of data without gaps
    are referred to as 'chunks'. For each chunk, determine best estimate of
    the first packet time, and then calculate time for each sample based on
    sampling rate -- assume no missing samples. Best estimate of start time
    for each chunk is determined by taking median (across all packets in that
    chunk) of the offset between delta PacketGenTime and expected time to
    have elapsed (as a function of sampling rate and number of samples
    per packet).

    Input:
    (1) Data table output from createTimeDomainTable, createAccelTable,
    etc (e.g. data originating from RawDataTD.json or RawDataAccel.json)
    (2) shortGaps_systemTick: Flag '0' or '1'. '1' indicates that systemTick
    should be used for chunk time alignment if gaps < 6 sec. Default is '0' -
    use packetGenTime to align all chunks

    Output: Same as input table, with additional column of 'DerivedTimes' for
    each sample
    """
    
    # Pull out info for each packet
    indices_of_timestamps = np.where(~np.isnan(input_data_table['timestamp']))[0]
    data_table_original = input_data_table.iloc[indices_of_timestamps]
    
    # Identify packets for rejection
    print('Identifying and removing bad packets')
    
    # Remove any packets with timestamp that are more than 24 hours from median timestamp
    median_timestamp = np.median(data_table_original['timestamp'])
    num_secs = 24 * 60 * 60
    bad_date_packets = np.union1d(np.where(data_table_original['timestamp'] > median_timestamp + num_secs)[0],
                                  np.where(data_table_original['timestamp'] < median_timestamp - num_secs)[0])
    
    # Negative PacketGenTime
    packet_indices_neg_gen_time = np.where(data_table_original['PacketGenTime'] <= 0)[0]
    
    # Identify PacketGenTime that is more than 2 seconds different from
    # timestamp (indicating outlier PacketGenTime)
    # Find first packet which does not have a negative packetGenTime
    temp_start_index = np.min(np.setdiff1d(np.arange(len(indices_of_timestamps)), packet_indices_neg_gen_time))
    
    elapsed_timestamp = data_table_original['timestamp'].iloc[temp_start_index:].values - data_table_original['timestamp'].iloc[temp_start_index]
    elapsed_packet_gen_time = (data_table_original['PacketGenTime'].iloc[temp_start_index:].values - data_table_original['PacketGenTime'].iloc[temp_start_index]) / 1000
    time_difference = np.abs(elapsed_timestamp - elapsed_packet_gen_time)
    
    packet_indices_outlier_packet_gen_time = np.where(time_difference > 5)[0]
    
    # Consecutive packets with identical dataTypeSequence and systemTick;
    # identify the second packet for removal; identify the first
    # instance of these duplicates below
    duplicate_first_index = np.intersect1d(np.where(np.diff(data_table_original['dataTypeSequence']) == 0),
                                           np.where(np.diff(data_table_original['systemTick']) == 0))
    
    # Identify packetGenTimes that go backwards in time by more than 500ms; may overlap with negative PacketGenTime
    packet_gen_time_diffs = np.diff(data_table_original['PacketGenTime'])
    diff_indices = np.where(packet_gen_time_diffs < -500)[0]
    
    # Need to remove [diffIndices + 1], but may also need to remove subsequent
    # packets. Automatically remove the next packet [diffIndices + 2], as this is easier than
    # trying to confirm there is enough time to assign to samples without
    # causing overlap.
    # Remove at most 6 adjacent packets (to prevent large un-needed
    # packet rejection driven by positive outliers)
    num_packets = data_table_original.shape[0]
    indices_back_in_time = []
    for i_index in range(len(diff_indices)):
        counter = 3  # Automatically removing two packets, start looking at the third
        
        # Check if next packet indices exists in the recording
        if (diff_indices[i_index] + 1) <= num_packets:
            indices_back_in_time.append(diff_indices[i_index] + 1)
        if (diff_indices[i_index] + 2) <= num_packets:
            indices_back_in_time.append(diff_indices[i_index] + 2)
        
        # If there are more packets after this, check if they need to also be
        # removed
        while (counter <= 6) and ((diff_indices[i_index] + counter) <= num_packets) and \
              (data_table_original['PacketGenTime'].iloc[diff_indices[i_index] + counter] < 
               data_table_original['PacketGenTime'].iloc[diff_indices[i_index]]):
            
            indices_back_in_time.append(diff_indices[i_index] + counter)
            counter += 1
    
    # Collect all packets to remove
    packets_to_remove = np.unique(np.concatenate((bad_date_packets, packet_indices_neg_gen_time,
                                                  duplicate_first_index + 1, indices_back_in_time,
                                                  packet_indices_outlier_packet_gen_time)))
    
    # Remove packets identified above for rejection
    packets_to_keep = np.setdiff1d(np.arange(data_table_original.shape[0]), packets_to_remove)
    data_table = data_table_original.iloc[packets_to_keep]
    
    del data_table_original
    
    # Identify gaps -- start with most obvious gaps, then become more refined
    
    # Change in sampling rate should start a new chunk; identify indices of
    # last packet of a chunk
    if len(data_table['samplerate'].unique()) != 1:
        indices_change_fs = np.where(np.diff(data_table['samplerate']) != 0)[0]
    else:
        indices_change_fs = []
    
    max_fs = data_table['samplerate'].max()
    
    # Timestamp difference from previous packet not 0 or 1 -- indicates gap or
    # other abnormality; these indices indicate the end of a continuous chunk
    indices_timestamp_flagged = np.intersect1d(np.where(np.diff(data_table['timestamp']) != 0),
                                               np.where(np.diff(data_table['timestamp']) != 1))
    
    # Instances where dataTypeSequence doesn't iterate by 1; these indices
    # indicate the end of a continuous chunk
    indices_data_type_sequence_flagged = np.intersect1d(np.where(np.diff(data_table['dataTypeSequence']) != 1),
                                                        np.where(np.diff(data_table['dataTypeSequence']) != -255))
    
    # Lastly, check systemTick to ensure there are no packets missing.
    # Determine delta time between systemTicks in adjacent packets.
    # Exclude first packet from those to test, because need to compare times from
    # 'previous' to 'current' packets; diff_systemTick written to index of
    # second packet associated with that calculation
    num_packets = data_table.shape[0]
    
    diff_system_tick = np.full(num_packets, np.nan)
    for i_packet in range(1, num_packets):
        diff_system_tick[i_packet] = np.mod((data_table['systemTick'].iloc[i_packet] + (2**16) - 
                                             data_table['systemTick'].iloc[i_packet - 1]), 2**16)
    
    # Expected elapsed time for each packet, based on sample rate and number of
    # samples per packet; in units of systemTick (1e-4 seconds)
    expected_elapsed = data_table['packetsizes'] * (1 / data_table['samplerate']) * 1e4
    
    # If diff_systemTick and expectedElapsed differ by more than the larger of 50% of expectedElapsed OR 100ms,
    # flag as gap
    cutoff_values = np.maximum(0.5 * expected_elapsed, np.full_like(expected_elapsed, 1000))
    indices_system_tick_flagged = np.where(np.abs(expected_elapsed[1:] - diff_system_tick[1:]) > cutoff_values[1:])[0]
    
    # All packets flagged as end of continuous chunks
    all_flagged_indices = np.unique(np.concatenate((indices_change_fs, indices_timestamp_flagged,
                                                    indices_data_type_sequence_flagged, indices_system_tick_flagged)))
    
    print('Chunking data')
    
    # Determine indices of packets which correspond to each data chunk
    if len(all_flagged_indices) > 0:
        counter = 1
        chunk_indices = [None] * (len(all_flagged_indices) + 1)
        for i_chunk in range(len(all_flagged_indices)):
            if i_chunk == 0:
                chunk_indices[counter - 1] = np.arange(all_flagged_indices[0] + 1)
                current_start_index = 0
                counter += 1
                # Edge case: Only one flagged index; automatically create
                # second chunk to end of data
                if len(all_flagged_indices) == 1:
                    chunk_indices[counter - 1] = np.arange(all_flagged_indices[current_start_index] + 1, num_packets)
            elif i_chunk == len(all_flagged_indices) - 1:
                chunk_indices[counter - 1] = np.arange(all_flagged_indices[current_start_index] + 1, all_flagged_indices[current_start_index + 1] + 1)
                chunk_indices[counter] = np.arange(all_flagged_indices[current_start_index + 1] + 1, num_packets)
            else:
                chunk_indices[counter - 1] = np.arange(all_flagged_indices[current_start_index] + 1, all_flagged_indices[current_start_index + 1] + 1)
                current_start_index += 1
                counter += 1
    else:
        # No identified missing packets, all packets in one chunk
        chunk_indices = [np.arange(num_packets)]
    
    # Two types of chunks -- those which come after gaps < 6 seconds (as determined by
    # timestamp) and those which come after gaps > 6 seconds (potential for complete
    # roll-over of systemTick). For chunks which follow a gap < 6 seconds,
    # there are two options for processing (flagged using shortGaps_systemTick)
    # - '1' indicates use systemTick to continue calculation of DerivedTime after
    # the gap (rather than creating a new correctedAlignTime for that chunk); still honor 1/Fs
    # spacing of DerivedTime. Default (and shortGaps_systemTick = 0) is all chunks
    # use a new correctedAlignTime. In some recordings, using systemTick led to
    # increasing offset between datastreams across recordings, presumably
    # because of inaccuracy of the systemTick clock. However some users found the
    # systemTick method to provide better accuracy. Note -- if new chunk is created because of change
    # in sampling rate, calculate a new correctedAlignTime for that chunk
    
    num_chunks = len(chunk_indices)
    
    # If using the systemTick method for gaps < 6 seconds
    if short_gaps_system_tick == 1:
        chunks_with_timing_from_previous = []
        # Only need to do this calculation if more than 1 chunk
        if num_chunks > 1:
            # Get timestamps of first and last packet in chunks to calculate time gaps
            indices_first_packet = [chunk[0] for chunk in chunk_indices]
            indices_last_packet = [chunk[-1] for chunk in chunk_indices]
            
            timestamps_first_packet = data_table['timestamp'].iloc[indices_first_packet].values
            timestamps_last_packet = data_table['timestamp'].iloc[indices_last_packet].values
            timestamp_gaps = timestamps_first_packet[1:] - timestamps_last_packet[:-1]  # in seconds
            
            # If the gap in timestamp between packets is < 6 seconds, flag this packet;
            # calculate elapsed time in systemTick
            
            # iTimegap + 1 is the index of the chunk which does not need a new
            # correctedAlignTime calculated (aka chunksWithTimingFromPrevious)
            elapsed_system_tick = np.full(len(timestamp_gaps), np.nan)
            
            for i_timegap in range(len(timestamp_gaps)):
                # Check if timegap is < 6 seconds and if this chunk was not created
                # because of change in sampling rate
                if timestamp_gaps[i_timegap] < 6 and indices_last_packet[i_timegap] not in indices_change_fs:
                    chunks_with_timing_from_previous.append(i_timegap + 1)
                    system_tick_first_packet = data_table['systemTick'].iloc[indices_first_packet[i_timegap + 1]]
                    system_tick_preceeding = data_table['systemTick'].iloc[indices_last_packet[i_timegap]]
                    
                    # Need to use calculateDeltaSystemTick in order to handle situations when
                    # systemTick rollover occurred
                    elapsed_system_tick[i_timegap] = calculate_delta_system_tick(system_tick_preceeding, system_tick_first_packet)
    
    # Loop through each chunk to determine offset to apply (as determined by
    # average difference between packetGenTime and expectedElapsed) --
    # calculated for all chunks here, will subsequently only apply error for
    # chunks which are preceeded by gaps >= 6 seconds if shortGaps_systemTick
    # == 1
    
    print('Determining start time of each chunk')
    
    # PacketGenTime in ms; convert difference to 1e-4 seconds, units of
    # systemTick and expectedElapsed
    diff_packet_gen_time = np.concatenate(([1], np.diff(data_table['PacketGenTime']) * 1e1))  # multiply by 1e1 to convert to 1e-4 seconds
    single_packet_chunks = []
    median_error = np.full(num_chunks, np.nan)
    
    for i_chunk in range(num_chunks):
        current_timestamp_indices = chunk_indices[i_chunk]
        
        # Chunks must have at least 2 packets in order to have a valid
        # diff_PacketGenTime -- thus if chunk only one packet, it must be
        # identified. These chunks can remain if the timeGap before is < 6
        # seconds, but must be excluded if the timeGap before is >= 6 seconds
        # or error set to zero
        
        if len(current_timestamp_indices) == 1:
            # For chunks with only 1 packet, zero error
            single_packet_chunks.append(i_chunk)
            median_error[i_chunk] = 0
        else:
            # Always exclude the first packet of the chunk, because don't have an
            # accurate diff_systemTick value for this first packet
            current_timestamp_indices = current_timestamp_indices[1:]
            
            # Differences between adjacent PacketGenTimes (in units of 1e-4
            # seconds)
            error = expected_elapsed.iloc[current_timestamp_indices.astype(int)] - diff_packet_gen_time[current_timestamp_indices.astype(int)]
            median_error[i_chunk] = np.median(error)
    
    # Create corrected timing for each chunk
    counter = 1
    counter_recalculated_from_packet_gen_time = 0
    corrected_align_time = np.full(num_chunks, np.nan)
    
    for i_chunk in range(num_chunks):
        if short_gaps_system_tick == 1 and i_chunk in chunks_with_timing_from_previous:
            # Determine amount of cumulative time since the previous
            # packet's correctedAlignTime -- add this cumulative time to
            # the previous packet's correctedAlignTime in order to
            # calculate the current packet's correctedAlignTime
            
            # elapsed_systemTick accounts for time from last packet in the
            # preceeding chunk to the first packet in the current chunk
            
            # otherTime_previousChunk accounts for time from fist packet to
            # last packet in the previous chunk; do this as a function of
            # number of samples and Fs (these two chunks will have the same
            # Fs, as enforced above)
            fs_previous_chunk = data_table['samplerate'].iloc[chunk_indices[i_chunk - 1][0]]
            
            all_packet_sizes_previous_chunk = data_table['packetsizes'].iloc[chunk_indices[i_chunk - 1]]
            
            # We just need to account for time from samples from packets two to end of the
            # previous chunk (in ms)
            other_time_previous_chunk = np.sum(all_packet_sizes_previous_chunk[1:]) * (1 / fs_previous_chunk) * 1000
            
            corrected_align_time[counter - 1] = corrected_align_time[counter - 2] + \
                                                 (elapsed_system_tick[i_chunk - 1] * 1e-1) + other_time_previous_chunk
            counter += 1
        else:
            align_time = data_table['PacketGenTime'].iloc[int(chunk_indices[i_chunk][0])]
            # alignTime in ms; medianError in units of systemTick
            corrected_align_time[counter - 1] = align_time + median_error[i_chunk] * 1e-1
            # Development Note: The medianError calculated and applied here
            # only includes samples within an original chunk; thus, if
            # there are two chunks with < 6 second gap, only the error
            # calculated from the first chunk will be used to create the
            # correctedAlignTime
            
            # Adding error above because we assume the expectedElapsed time (function of
            # sampling rate and number of samples in packet) represents the
            # correct amount of elapsed time. We calculated the median difference
            # between the expected elapsed time according to the packet size
            # and the diff PacketGenTime. The number of time units will be
            # negative if the diff PacketGenTime is consistently larger than
            # the expected elapsed time, so adding removes the bias.
            # The alternative would be if we thought PacketGenTime was a more
            # accurate representation of time, then we would want to subtract the value in medianError.
            counter += 1
            counter_recalculated_from_packet_gen_time += 1
    
    # Print metrics to command window
    print(f"Number of chunks: {num_chunks}")
    print(f"Number of chunks with time calculated from PacketGenTime: {counter_recalculated_from_packet_gen_time}")
    
    # At this point, possible that all chunks have been removed - check for
    # this and only proceed with processing if chunks remain
    
    if 'corrected_align_time' in locals():
        # Indices in chunkIndices correspond to packets in dataTable.
        # CorrectedAlignTime corresponds to first packet for each chunk in
        # chunkIndices. Remove chunks identified above
        
        # correctedAlignTime is shifted slightly to keep times exactly aligned to
        # sampling rate; use maxFs for alignment points. In other words, take first
        # correctedAlignTime, and place all other correctedAlignTimes at multiples of
        # (1/Fs)
        delta_time = 1 / max_fs * 1000  # in milliseconds
        
        print('Shifting chunks to align with sampling rate')
        multiples = np.floor(((corrected_align_time - corrected_align_time[0]) / delta_time) + 0.5)
        corrected_align_time_shifted = corrected_align_time[0] + (multiples * delta_time)
        
        # Note: In some instances, the correctedAlignTime_shifted times will go
        # backwards in time. We do not exclude at this point (because we do not
        # want to exclude ALL data in that chunk). Below, we remove the
        # offending samples (which should be group offending in sets of ~packet
        # size)
        
        # Full form data table
        output_data_table = input_data_table.copy()
        del input_data_table
        
        # Remove packets and samples identified above as lacking proper metadata
        samples_to_remove = []
        # If first packet is included, collect those samples separately
        if 1 in packets_to_remove:
            samples_to_remove = np.arange(1, indices_of_timestamps[0] + 1)
            packets_to_remove = np.setdiff1d(packets_to_remove, [1])
        
        # Loop through all other packetsToRemove
        to_remove_start = indices_of_timestamps[packets_to_remove - 1] + 1
        to_remove_stop = indices_of_timestamps[packets_to_remove]
        for i_packet in range(len(packets_to_remove)):
            # samples_to_remove.extend(np.arange(to_remove_start[i_packet], to_remove_stop[i_packet] + 1))
            samples_to_remove = np.concatenate((samples_to_remove, np.arange(to_remove_start[i_packet], to_remove_stop[i_packet] + 1)))
        
        output_data_table.drop(output_data_table.index[samples_to_remove], inplace=True)
        
        # Indices referenced in chunkIndices can now be mapped back to timeDomainData
        # using indicesOfTimestamps_cleaned
        indices_of_timestamps_cleaned = np.where(~np.isnan(output_data_table['timestamp']))[0]
        
        # Map the chunk start/stop times back to samples
        chunk_packet_start = np.zeros(len(chunk_indices), dtype=int)
        chunk_sample_start = np.zeros(len(chunk_indices), dtype=int)
        chunk_sample_end = np.zeros(len(chunk_indices), dtype=int)
        
        for i_chunk in range(len(chunk_indices)):
            current_packets = chunk_indices[i_chunk]
            chunk_packet_start[i_chunk] = indices_of_timestamps_cleaned[current_packets[0]]
            if current_packets[0] == 0:  # First packet, thus take first sample
                chunk_sample_start[i_chunk] = 0
            else:
                chunk_sample_start[i_chunk] = indices_of_timestamps_cleaned[current_packets[0] - 1] + 1
            chunk_sample_end[i_chunk] = indices_of_timestamps_cleaned[current_packets[-1]]
        
        print('Creating derivedTime for each sample')
        
        # Use correctedAlignTime and sampling rate to assign each included sample a
        # derivedTime
        
        # Initialize DerivedTime
        derived_time = np.full(output_data_table.shape[0], np.nan)
        for i_chunk in range(len(chunk_indices)):
            # Display status
            if i_chunk > 0 and i_chunk % 1000 == 0:
                print(f"Currently on chunk {i_chunk} of {len(chunk_indices)}")
            
            # Assign derivedTimes to all samples (from before first packet time to end) -- all same
            # sampling rate
            current_fs = output_data_table['samplerate'].iloc[chunk_packet_start[i_chunk]]
            elapsed_time_before = (chunk_packet_start[i_chunk] - chunk_sample_start[i_chunk]) * (1000 / current_fs)
            elapsed_time_after = (chunk_sample_end[i_chunk] - chunk_packet_start[i_chunk]) * (1000 / current_fs)
            
            derived_time[chunk_sample_start[i_chunk]:chunk_sample_end[i_chunk] + 1] = \
                np.arange(corrected_align_time_shifted[i_chunk] - elapsed_time_before,
                          corrected_align_time_shifted[i_chunk] + elapsed_time_after + 1000 / current_fs,
                          1000 / current_fs)
        
        # Identify samples which have DerivedTime that goes backwards in time -
        # flag those samples for removal
        current_value = derived_time[0]
        next_value = derived_time[1]
        indices_to_remove = []
        
        for i_sample in range(len(derived_time) - 1):
            if current_value < next_value:
                # Checks if the current value is smaller than the next value - if
                # yes, keep this value and iterate currentValue and nextValue
                # for next loop
                current_value = derived_time[i_sample + 1]
            else:
                # if the currentValue is not smaller than nextValue, collect
                # the index of nextValue to later remove
                indices_to_remove.append(i_sample + 1)
                # currentValue does not iterate
            
            # Iterate nextValue regardless of condition above
            if i_sample < len(derived_time) - 2:
                next_value = derived_time[i_sample + 2]
        
        # Check to ensure that the same DerivedTime was not assigned to multiple
        # samples; if yes, flag the second instance for removal; note: in numpy, nans
        # are not equal
        if len(derived_time) != len(np.unique(derived_time)):
            _, unique_indices = np.unique(derived_time, return_index=True)
            duplicate_indices = np.setdiff1d(np.arange(len(derived_time)), unique_indices)
        else:
            duplicate_indices = []
        
        # Combine indicesToRemove and duplicateIndices
        combined_to_remove = np.union1d(indices_to_remove, duplicate_indices)
        
        # All samples which do not have a derivedTime should be removed from final
        # data table, along with those with duplicate derivedTime values
        print('Cleaning up output table')
        output_data_table['DerivedTime'] = derived_time
        rows_to_remove = np.concatenate((np.where(np.isnan(derived_time))[0], combined_to_remove))
        output_data_table.drop(output_data_table.index[rows_to_remove], inplace=True)
        
        # Make timing/metadata variables consistent across data streams
        output_data_table = output_data_table[['DerivedTime', 'timestamp', 'systemTick', 'PacketGenTime', 'PacketRxUnixTime',
                                               'dataTypeSequence', 'samplerate', 'packetsizes'] +
                                              [col for col in output_data_table.columns if col not in ['DerivedTime', 'timestamp',
                                                                                                       'systemTick', 'PacketGenTime',
                                                                                                       'PacketRxUnixTime', 'dataTypeSequence',
                                                                                                       'samplerate', 'packetsizes']]]
    else:
        output_data_table = pd.DataFrame()
    
    return output_data_table


def calculate_delta_system_tick(system_tick_preceeding, system_tick_first_packet):
    """
    Helper function to calculate the delta system tick between two system tick values,
    handling potential rollover of system tick.
    """
    delta_system_tick = np.mod((system_tick_first_packet + (2**16) - system_tick_preceeding), 2**16)
    return delta_system_tick