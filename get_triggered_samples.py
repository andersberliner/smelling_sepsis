    def get_triggered_samples(self, sample_window, triggers):
        left_window_edge, right_window_edge = sample_window
        number_of_samples = right_window_edge - left_window_edge + 1
        samples = RecordSet(self, deep_copy_data=False)
        for analyte_index in range(self.number_of_analytes):
            analyte_triggers = triggers[analyte_index]
            analyte = self.analyte_records[analyte_index]
            sample_analyte = samples.analyte_records[analyte_index]
            for record_index in range(len(analyte)):
                record = analyte[record_index]
                sample_record = sample_analyte[record_index]

                # Create 'Data' field.
                feature_times = record['Data']
                number_of_features, number_of_times = feature_times.shape
                sample_feature_times = numpy.empty((number_of_features, number_of_samples), dtype='float')
                sample_record['Data'] = sample_feature_times

                # Trigger sample.
                trigger = analyte_triggers[record_index]
                sample_record['Trigger'] = trigger
                number_of_left_blanks = max(0, -(trigger + left_window_edge))
                number_of_right_blanks = max(0, trigger + right_window_edge - (number_of_times-1))                                
                for feature in range(number_of_features):
                    for sample in range(number_of_left_blanks):
                        sample_feature_times[feature, sample] = 0
                    sample_index = trigger + left_window_edge + number_of_left_blanks
                    for sample in range(number_of_left_blanks, number_of_samples - number_of_right_blanks):
                        sample_feature_times[feature, sample] = feature_times[feature, sample_index]
                        sample_index += 1
                    for sample in range(number_of_samples - number_of_right_blanks, number_of_samples):
                        sample_feature_times[feature, sample] = 0
        samples.x_axis = range(left_window_edge, right_window_edge+1)
        return samples
        
        
        
    sample_triggers = second_derivatives.get_sample_triggers(trigger_function)
    
    # Sample a window centered on the trigger for each trial.
    sample_radius = 21
    sample_interval = 3
#    sample_interval = 1
    sample_window = [-sample_radius, sample_radius]
#    sample_window = [-9, sample_radius]
    difference_samples = differences.get_triggered_samples(sample_window, sample_triggers)
    first_derivative_samples = first_derivatives.get_triggered_samples(sample_window, sample_triggers)
    second_derivative_samples = second_derivatives.get_triggered_samples(sample_window, sample_triggers)

    slices = [-sample_window[0] + t for t in range(sample_window[0], sample_window[1]+1, sample_interval)]
    first_derivative_sample_time_slices = first_derivative_samples.get_slices(slices)
    second_derivative_sample_time_slices = second_derivative_samples.get_slices(slices)
    number_of_time_slices = len(slices)
