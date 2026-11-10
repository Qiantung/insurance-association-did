import unittest
from research import synthetic_panel,validate,fit_baseline

class PanelChecks(unittest.TestCase):
    def test_duplicate_rejected(self):
        import pandas as pd
        df=synthetic_panel()
        with self.assertRaises(ValueError):validate(pd.concat([df,df.iloc[:1]]))
    def test_status_change_rejected(self):
        df=synthetic_panel();df.loc[0,'treated']=0
        with self.assertRaises(ValueError):validate(df)
    def test_known_synthetic_effect(self):
        df=synthetic_panel();self.assertEqual(validate(df)['rows'],440)
        model=fit_baseline(df)
        self.assertLess(abs(model.params['did']-2),.7)

if __name__=='__main__':unittest.main()
