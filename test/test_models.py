import unittest

import fmax as fm

class TestModels(unittest.TestCase):
    def setUp(self):
        pass
      
    def test_fitting_gaussian_model(self):
        # Generate sample data
        DATA_LEN = 30
        FCAST_LEN = 10
        random_fn = fm.get_random_fn(
            n_periods = DATA_LEN, 
            past_obs = None,
            attempts = "gaussian",
            kind = 'min',
            )
        TRUE_MU = 10
        TRUE_SIGMA = 1
        record_data = random_fn(point = {'mu' : TRUE_MU, 
                                         'sigma' : TRUE_SIGMA}
                                )
        current_record = record_data[-1]
        
        self.assertEqual(len(record_data), DATA_LEN)
        
        # Create model
        model = fm.ForecastModel(
                      record_data, 
                      time_index=None, 
                      fcast_len = FCAST_LEN,
                      kind="max", 
                      attempts="gaussian", 
                      train="all"
                      )
        
        # Get conditional distribution of forecast
        trace = model.fit()
        
        mean_posterior_mu = trace['posterior']['mu'].mean()
        self.assertAlmostEqual(mean_posterior_mu, TRUE_MU, delta=2)
        
        mean_posterior_forecast = trace['posterior']['running_max'].mean(axis=0)
        
        self.assertEqual(mean_posterior_forecast.shape, (DATA_LEN + FCAST_LEN,))
        self.assertEqual(mean_posterior_forecast[:DATA_LEN], record_data)
        for mean_pred in mean_posterior_forecast[DATA_LEN:]:
          self.assertGreaterOrEqual(mean_pred, current_record)
        
        # Get posterior predictive
        post_pred = model.sample_posterior_predictive()
        mean_post_pred = post_pred['running_max'].mean(axis=0)
        self.assertEqual(mean_post_pred.shape, (DATA_LEN + FCAST_LEN,))
        