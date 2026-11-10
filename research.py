"""Research-in-progress infrastructure. Default demo uses synthetic data only.

Common-adoption TWFE demonstration; not a staggered-adoption estimator.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def synthetic_panel(seed=42,cities=40):
    rng=np.random.default_rng(seed);rows=[]
    for city in range(cities):
        treated=int(city<cities//2);city_effect=rng.normal(0,2)
        for year in range(2000,2011):
            x=rng.normal();did=treated*int(year>=2005)
            y=city_effect+.3*(year-2000)+.7*x+2*did+rng.normal()
            rows.append((f'SYNTH_{city:03}',year,treated,x,y,True))
    return pd.DataFrame(rows,columns=['city','year','treated','control','outcome','synthetic'])

def validate(df):
    required={'city','year','treated','control','outcome'}
    if not required<=set(df):raise ValueError(f'Missing columns: {sorted(required-set(df))}')
    if df[list(required)].isna().any().any():raise ValueError('Missing data; resolve before regression.')
    if df.duplicated(['city','year']).any():raise ValueError('Duplicate city-year observations')
    if not set(df.treated.unique())=={0,1}:raise ValueError('treated must explicitly code 1=treated, 0=control.')
    if df.groupby('city').treated.nunique().max()!=1:raise ValueError('Demo requires time-invariant treated status.')
    for col in ['year','treated','control','outcome']:
        if not pd.api.types.is_numeric_dtype(df[col]) or not np.isfinite(df[col]).all():raise ValueError(f'Invalid numeric column: {col}')
    if not np.equal(df.year,np.floor(df.year)).all():raise ValueError('Year must be integral')
    if df.groupby('treated').city.nunique().min()<2:raise ValueError('Too few city clusters per treatment arm')
    return {'rows':len(df),'cities':df.city.nunique(),'years':sorted(int(y) for y in df.year.unique()),
            'balanced':bool(df.groupby('city').year.nunique().eq(df.year.nunique()).all())}

def fit_baseline(df,policy_year=2005):
    validate(df);data=df.copy();data['did']=data.treated*(data.year>=policy_year).astype(int)
    for group in [0,1]:
        years=data[data.treated==group].year
        if not (years.lt(policy_year).any() and years.ge(policy_year).any()):raise ValueError('Each arm needs pre/post observations')
    # City/time fixed effects absorb the treated and post main effects.
    return smf.ols('outcome ~ did + control + C(city) + C(year)',data=data).fit(
        cov_type='cluster',cov_kwds={'groups':data.city})

def event_study(df,policy_year=2005):
    validate(df);data=df.copy();relative=sorted(int(y-policy_year) for y in data.year.unique())
    if -1 not in relative:raise ValueError('Reference year t=-1 must be observed')
    terms=[];lookup={}
    for rel in relative:
        if rel==-1:continue
        name=f'event_m{abs(rel)}' if rel<0 else f'event_p{rel}'
        data[name]=data.treated*((data.year-policy_year)==rel).astype(int);terms.append(name);lookup[name]=rel
    model=smf.ols('outcome ~ control + C(city) + C(year) + '+' + '.join(terms),data=data).fit(
        cov_type='cluster',cov_kwds={'groups':data.city})
    ci=model.conf_int();rows=[{'event_time':lookup[n],'estimate':float(model.params[n]),
                              'ci_low':float(ci.loc[n,0]),'ci_high':float(ci.loc[n,1])} for n in terms]
    pre=[n for n in terms if lookup[n]<-1]
    pre_p=float(model.wald_test(', '.join(n+' = 0' for n in pre),scalar=True).pvalue) if pre else None
    return rows,pre_p

def placebo(df,repetitions=100,seed=42):
    if repetitions<1:raise ValueError('Positive repetitions required')
    validate(df);rng=np.random.default_rng(seed)
    city_status=df.groupby('city',sort=True).treated.first();estimates=[]
    # Permute city labels, never individual panel rows, to preserve panel assignment.
    for _ in range(repetitions):
        perm=dict(zip(city_status.index,rng.permutation(city_status.to_numpy())))
        data=df.copy();data['treated']=data.city.map(perm);estimates.append(float(fit_baseline(data).params['did']))
    return estimates

def run(df,repetitions=100):
    audit=validate(df);model=fit_baseline(df);events,pre_p=event_study(df);draws=placebo(df,repetitions)
    coefficient=float(model.params['did']);ci=model.conf_int().loc['did'].tolist()
    return {'status':'Research in progress; demonstration only','data_audit':audit,
            'synthetic_only':bool('synthetic' in df and df.synthetic.eq(True).all()),
            'baseline':{'coefficient':coefficient,'clustered_se':float(model.bse['did']),
                        'p_value':float(model.pvalues['did']),'ci_95':ci},
            'event_study':events,'pretrend_joint_p':pre_p,
            'placebo':{'repetitions':repetitions,'seed':42,'coefficients':draws,
                       'permutation_tail_fraction':float((1+np.sum(np.abs(draws)>=abs(coefficient)))/(repetitions+1))},
            'limitations':['Failure to reject pretrends does not establish parallel trends.',
                          'Label permutation requires an assignment/exchangeability justification for real data.',
                          'Common-adoption TWFE is not sufficient for heterogeneous staggered adoption.',
                          'Synthetic estimates do not support any real-world policy claim.']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--placebos',type=int,default=100);args=p.parse_args()
    df=pd.read_csv(args.input) if args.input else synthetic_panel()
    result=run(df,args.placebos);args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    if args.input is None:df.to_csv(args.output/'synthetic_panel.csv',index=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ['event_study','placebo']},indent=2))
