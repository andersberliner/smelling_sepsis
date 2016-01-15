    
    # Define the trigger feature on which to center the sample window.
    #
    # Choose two features, A and B.
    # Feature B is for P.aeruginosa, feature A is for everything else.
    # If max(A) > threshold, center sample on max(A); otherwise center on max(B).
    def trigger_function(record_set, trial):
        def min_max_trigger(trial, feature, threshold):
            min_feature = numpy.min(trial[feature])
            max_feature = numpy.max(trial[feature])
            min_trigger = numpy.Infinity
            max_trigger = numpy.Infinity
            if min_feature < -threshold:
                min_trigger = numpy.argmin(trial[feature])
            if max_feature > threshold:
                max_trigger = numpy.argmax(trial[feature])
            trigger = min(min_trigger, max_trigger)
            if trigger == numpy.Infinity:
                trigger = 0
            return trigger
        
        feature = record_set.get_feature_index('26B')
        trigger = min_max_trigger(trial, feature, 0.02)
        if trigger > 0:
            return trigger
        feature = record_set.get_feature_index('11R')
        trigger = min_max_trigger(trial, feature, 0.02)
        if trigger > 0:
            return trigger
        feature = record_set.get_feature_index('45B')
        trigger = min_max_trigger(trial, feature, 0.04)
        if trigger > 0:
            return trigger
#        feature = record_set.get_feature_index('23R')
#        trigger = min_max_trigger(trial, feature)
#        if trigger > -1:
#            return trigger
#        feature = record_set.get_feature_index('15B')
#        trigger = min_max_trigger(trial, feature)
#        if trigger > -1:
#            return trigger
        feature = record_set.get_feature_index('36B')
        trigger = min_max_trigger(trial, feature, 0.1)
        return trigger
        
        feature_A = record_set.get_feature_index('30R')
#        feature_B = record_set.get_feature_index('36B')
#        feature_B = record_set.get_feature_index('9B')
        feature_B = record_set.get_feature_index('11B')
        abs_feature_A = numpy.abs(trial[feature_A])
        abs_feature_B = numpy.abs(trial[feature_B])
        max_abs_feature_A = numpy.max(abs_feature_A)
        max_abs_feature_B = numpy.max(abs_feature_B)
        if max_abs_feature_A > max_abs_feature_B:
            trigger_feature = abs_feature_A
        else:
            trigger_feature = abs_feature_B
        trigger = numpy.argmax(trigger_feature)
        if numpy.max(trigger_feature) < 0.02:
            trigger = -1
        return trigger

    # Get sample triggers for each trial from peaks in the first derivative.
#    sample_triggers = first_derivatives.get_sample_triggers(trigger_function)
    sample_triggers = second_derivatives.get_sample_triggers(trigger_function)
