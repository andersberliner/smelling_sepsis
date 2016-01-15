    def get_derivatives(self, order, dimension='time'):
        axis = self._get_axis(dimension)
        derivatives = RecordSet(self)
        for analyte_index in range(derivatives.number_of_analytes):
            analyte = self.analyte_records[analyte_index]
            derivative_analyte = derivatives.analyte_records[analyte_index]
            for record_index in range(len(analyte)):
                feature_times = analyte[record_index]['Data']
                derivative_analyte[record_index]['Data'] = numpy.diff(feature_times, order, axis)
        numpy.delete(derivatives.x_axis, -1)
        if dimension == 'feature':
            derivatives.number_of_features -= order
        return derivatives
        
        
    # smoothed = get_smoothed(sigma)
    #
    # - Convolves all time series with a Gaussian filter with standard deviation <sigma>.
    # - <smoothed> is a RecordSet.
    # - <sigma> is the standard deviation of the Gaussian filter.
    def get_smoothed(self, sigma, dimension='time'):
        axis = self._get_axis(dimension)
        smoothed = RecordSet(self)
        for analyte_index in range(smoothed.number_of_analytes):
            analyte = self.analyte_records[analyte_index]
            smoothed_analyte = smoothed.analyte_records[analyte_index]
            for record_index in range(len(analyte)):
                feature_times = analyte[record_index]['Data']
                smoothed_analyte[record_index]['Data'] = ndimage.gaussian_filter1d(feature_times, sigma, axis=axis)
        return smoothed
        
        
