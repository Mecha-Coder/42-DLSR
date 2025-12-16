import numpy as np
import pandas as pd

# ____________________________________________________________________

def accuracy(predict, actual):
    """Correct prediction / total"""
    
    if len(predict) != len(actual):
        raise Exception("Incorrect shape")
    acc = (np.sum(predict == actual) / len(actual)) * 100
    print(f"Accuracy = {acc:.3}%")


# ____________________________________________________________________

def interpolate(sorted_num, percentile, size):
    """lower + (upper - lower) x frac"""
    pos = percentile * (size - 1)
    i = int(pos)  # lower index
    j = i + 1     # upper index

    if (pos == i):
        return sorted_num[i]
    
    frac = pos - i
    lower = sorted_num[i]
    upper = sorted_num[j]
    return lower + (upper - lower) * frac

# ____________________________________________________________________
        

class Describe:
    def __init__(t, df: pd.DataFrame):

        t.row_no = len(df)
        t.col_no = len(df.columns)
        t.column = list(df.columns)
        t.dtype  = []
        t.sample = []
        t.unique = []
        t.missing= []
        t.count  = []
        t.mean   = []
        t.std    = []
        t.min    = []
        t.p25    = []
        t.p50    = []
        t.p75    = []
        t.max    = []

        for c in t.column :
            missing_no = 0
            values = []

            for i in range(t.row_no):
                val = df[c][i]
                
                if pd.isna(val): missing_no += 1
                else           : values.append(val)
            
            values = np.array(values)
            
            # --------------------------------------
            n = values.size
            t.count.append(n)
            t.missing.append(missing_no)

            # --------------------------------------
            unique_vals = np.unique(values)
            ulen = unique_vals.size
            u1, u2 = unique_vals[:2]

            if (ulen < 2 or type(u1) != type(u2)): 
                raise Exception("Data quality issue")
            
            t.dtype.append(type(u1).__name__)
            t.sample.append(u1)
            t.unique.append(ulen)
        
            # --------------------------------------
            if (not pd.api.types.is_number(u1)):
                t.mean.append(np.nan)
                t.std.append(np.nan)
                t.min.append(np.nan)
                t.p25.append(np.nan)
                t.p50.append(np.nan)
                t.p75.append(np.nan)
                t.max.append(np.nan)
                continue
            
            # --------------------------------------
            mean_result = np.sum(values) / n
            std_result = np.sqrt(np.sum((values - mean_result) ** 2) / n)

            t.mean.append(mean_result)
            t.std.append(std_result)

            values.sort()
            t.min.append(values[0])
            t.p25.append(interpolate(values, 0.25, n))
            t.p50.append(interpolate(values, 0.5, n))
            t.p75.append(interpolate(values, 0.75, n))
            t.max.append(values[-1])

    def show(t):
        print(f"Data size: {t.row_no} rows x {t.col_no} columns\n")

        print(  f"{'column_name':<15}",
                f" | {'type':<8}",
                f" | {'sample':<12}",
                f" | {'count':<6}",
                f" | {'missing':<6}",
                f" | {'unique':<6}",
                f" | {'mean':<12}",
                f" | {'std':<12}",
                f" | {'min':<12}", 
                f" | {'25%':<12}",
                f" | {'50%':<12}",
                f" | {'75%':<12}",
                f" | {'max':<12}"
            )
        print("-" * 180)

        for i in range(t.col_no):
            column  = f"{t.column[i]}" 
            type    = f"{t.dtype[i]}"
            sample  = f"{t.sample[i]}"
            count   = f"{t.count[i]}"
            missing = f"{t.missing[i]}"
            unique  = f"{t.unique[i]}"            
            mean    = f"{t.mean[i]}"  if not np.isnan(t.mean[i]) else "\"\""
            std     = f"{t.std[i]}"   if not np.isnan(t.std[i]) else "\"\""
            min     = f"{t.min[i]}"   if not np.isnan(t.min[i]) else "\"\""
            p25     = f"{t.p25[i]}"   if not np.isnan(t.p25[i]) else "\"\""
            p50     = f"{t.p50[i]}"   if not np.isnan(t.p50[i]) else "\"\""
            p75     = f"{t.p75[i]}"   if not np.isnan(t.p75[i]) else "\"\""
            max     = f"{t.max[i]}"   if not np.isnan(t.max[i]) else "\"\""

            print(  f"{column[:15]:<15}",
                    f" | {type[:8]:<8}",
                    f" | {sample[:12]:<12}",
                    f" | {count[:6]:<6}",
                    f" | {missing[:6]:<6}",
                    f" | {unique[:6]:<6}",
                    f" | {mean[:12]:<12}",
                    f" | {std[:12]:<12}",
                    f" | {min[:12]:<12}", 
                    f" | {p25[:12]:<12}",
                    f" | {p50[:12]:<12}",
                    f" | {p75[:12]:<12}", 
                    f" | {max[:12]:<12}"
                )