from main import Index_no_constraints, Index_constraint_1p,  Index_constraint_5p, Index_constraint_10p, stoxx600_TRI
import pandas as pd


df = pd.DataFrame(stoxx600_TRI).merge(pd.DataFrame(Index_no_constraints), on="Date")
df = pd.DataFrame(df).merge(pd.DataFrame(Index_constraint_1p), on="Date")
df = df.merge(pd.DataFrame(Index_constraint_5p), on="Date")
df = df.merge(pd.DataFrame(Index_constraint_10p), on="Date")
df.columns = ["Stoxx600 Index", "Index no constraints", "Index_constraint_1p", "Index_constraint_5p", "Index_constraint_10p"]
df.to_excel("Index.xlsx", "Index", startrow=0)






