# הסבר מפורט לקובץ BO.py

הסבר שורה אחר שורה לקובץ `system/BO.py` - מודול אופטימיזציה בייסיאנית לפרמטרים.

---

## שורות 1-3: כותרות והסבר כללי

```python
# system/BO.py
# Bayesian Optimization module for parameter optimization
# Uses Gaussian Process model, loss function, and Acquisition Function (UCB)
```

**שורה 1**: שם הקובץ ונתיבו במערכת הקבצים.

**שורה 2**: תמצית הקובץ - מודול אופטימיזציה בייסיאנית לפרמטרים.

**שורה 3**: כלים עיקריים:
- Gaussian Process - מודל הסתברותי למדידות קודמות
- Loss function - פונקציית עלות להערכת ביצועים
- Acquisition Function (UCB - Upper Confidence Bound) - פונקציית רכישה לבחירת הנקודה הבאה לבדיקה

---

## שורות 5-10: ייבוא ספריות

```python
import os
import config
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound
```

**שורה 5**: `os` - פעולות מערכת קבצים (בדיקת קיום קבצים).

**שורה 6**: `config` - קובץ הגדרות מקומי עם פרמטרים גלובליים.

**שורה 7**: `pandas` - טיפול בנתונים טבלאיים (קריאה מ-CSV).

**שורה 8**: `numpy` - חישובים מספריים (בדיקת NaN).

**שורה 9**: `BayesianOptimization` - מחלקה לביצוע אופטימיזציה בייסיאנית.

**שורה 10**: `UpperConfidenceBound` - פונקציית רכישה המאזנת בין exploration (חקר) ל-exploitation (ניצול).

---

## שורות 13-119: פונקציה `train_optimizer`

### שורות 13-28: חתימת הפונקציה ותיעוד

```python
def train_optimizer(result_csv_path=None):
    """
    Trains the Bayesian Optimizer with prior data from result.csv.
    
    Args:
        result_csv_path (str, optional): Path to result.csv file.
                                        If None, uses config.RESULTS_CSV_FILE
    
    Returns:
        BayesianOptimization: Trained optimizer object
    
    Process:
        1. Loads result.csv with previous simulation results
        2. Extracts parameter values and corresponding objective function values
        3. Registers all data points with the optimizer
    """
```

**שורה 13**: הגדרת הפונקציה. פרמטר אופציונלי - נתיב לקובץ תוצאות. אם לא מועבר, ייעשה שימוש בערך מ-config.

**שורות 14-28**: תיעוד (docstring):
- תפקיד: אימון האופטימיזר עם נתונים קודמים
- פרמטרים: נתיב לקובץ CSV (אופציונלי)
- החזרה: אובייקט אופטימיזר מאומן
- תהליך: טוען CSV, מחלץ פרמטרים וערכי פונקציית מטרה, ומרשם נקודות נתונים באופטימיזר

### שורות 29-30: קביעת נתיב קובץ התוצאות

```python
    if result_csv_path is None:
        result_csv_path = config.RESULTS_CSV_FILE
```

**שורה 29**: בדיקה אם לא הועבר נתיב.

**שורה 30**: שימוש בנתיב המוגדר ב-config.

### שורות 32-34: בדיקת קיום קובץ

```python
    # Load results CSV
    if not os.path.exists(result_csv_path):
        raise FileNotFoundError(f"Results file not found: {result_csv_path}. Run initial simulations first.")
```

**שורה 32**: הערה.

**שורה 33**: בדיקה אם הקובץ קיים.

**שורה 34**: זריקת שגיאה אם הקובץ לא קיים, עם הודעה ברורה.

### שורות 36-39: טעינת CSV ובדיקת ריקות

```python
    df = pd.read_csv(result_csv_path)
    
    if len(df) == 0:
        raise ValueError("Results file is empty. Run initial simulations first.")
```

**שורה 36**: טעינת הקובץ ל-DataFrame של pandas.

**שורה 38**: בדיקה אם הטבלה ריקה.

**שורה 39**: זריקת שגיאה אם הטבלה ריקה.

### שורה 41: הודעת התקדמות

```python
    print(f"  -> Loaded {len(df)} prior data points from {result_csv_path}")
```

הדפסת מספר נקודות הנתונים שנטענו.

### שורות 43-49: המרת פרמטרים לפורמט pbounds

```python
    # Convert SWEEP_PARAMETERS to pbounds format for BayesianOptimization
    # config has: {'w_r': {'min': 350e-9, 'max': 450e-9, 'unit': 'm'}}
    # BayesianOptimization needs: {'w_r': (350e-9, 450e-9)}
    pbounds = {}
    param_names = list(config.SWEEP_PARAMETERS.keys())
    for param_name, param_config in config.SWEEP_PARAMETERS.items():
        pbounds[param_name] = (param_config['min'], param_config['max'])
```

**שורה 43**: הערה - המרה לפורמט הנדרש.

**שורות 44-45**: הסבר על ההבדל בין הפורמטים:
- config: מילון עם 'min', 'max', 'unit'
- BayesianOptimization: tuple של (min, max)

**שורה 46**: יצירת מילון ריק לפרמטרים.

**שורה 47**: רשימת שמות הפרמטרים מה-config.

**שורות 48-49**: לולאה שבונה את pbounds:
- כל פרמטר הופך ל-tuple של (min, max)

### שורות 51-52: יצירת פונקציית רכישה

```python
    # Create acquisition function with kappa from config
    utility = UpperConfidenceBound(kappa=config.BO_KAPPA)
```

**שורה 51**: הערה - יצירת פונקציית רכישה.

**שורה 52**: יצירת פונקציית UCB עם kappa מ-config:
- kappa גבוה: יותר exploration
- kappa נמוך: יותר exploitation

### שורות 54-62: יצירת אופטימיזר

```python
    # Create optimizer
    # Note: f=None because we use suggest() instead of maximize()
    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        random_state=42,
        acquisition_function=utility,
        verbose=2,
    )
```

**שורה 54**: הערה.

**שורה 55**: הסבר: f=None כי משתמשים ב-suggest() במקום maximize().

**שורה 56**: יצירת האופטימיזר:
- **f=None**: אין פונקציית מטרה (נשתמש ב-suggest())
- **pbounds**: גבולות הפרמטרים
- **random_state=42**: זרע אקראי לקביעות
- **acquisition_function=utility**: פונקציית הרכישה
- **verbose=2**: רמת פירוט גבוהה

### שורות 64-66: אתחול מונים

```python
    # Register each prior data point
    registered_count = 0
    skipped_count = 0
```

**שורה 64**: הערה - רישום נקודות נתונים.

**שורה 65**: מונה לנקודות שנרשמו בהצלחה.

**שורה 66**: מונה לנקודות שדולגו.

### שורות 68-113: לולאת רישום נקודות נתונים

```python
    for index, row in df.iterrows():
```

**שורה 68**: לולאה על כל שורה בטבלה.

#### שורות 70-76: חילוץ ערכי פרמטרים

```python
        # Extract parameter values
        try:
            params = {}
            for param_name in param_names:
                if param_name not in row:
                    print(f"  [WARNING] Parameter '{param_name}' not found in row {index}. Skipping.")
                    break
                params[param_name] = row[param_name]
```

**שורה 70**: הערה - חילוץ ערכי פרמטרים.

**שורה 71**: תחילת try-except לטיפול בשגיאות.

**שורה 72**: מילון ריק לפרמטרים של השורה.

**שורה 73**: לולאה על כל שם פרמטר.

**שורה 74**: בדיקה אם הפרמטר קיים בשורה.

**שורה 75**: הודעת אזהרה ודילוג על השורה.

**שורה 76**: יציאה מהלולאה הפנימית.

**שורה 77**: הוספת ערך הפרמטר למילון.

#### שורות 78-95: חילוץ alpha ו-v_pi_l

```python
            else:
                # Extract loss and vpil from results
                # Check for column names (may vary based on result.csv structure)
                alpha = None
                v_pi_l = None
                
                if 'loss_at_v_pi_dB_per_cm' in row:
                    alpha = row['loss_at_v_pi_dB_per_cm']
                elif 'loss_db' in row:
                    alpha = row['loss_db']
                elif 'alpha' in row:
                    alpha = row['alpha']
                
                if 'v_pi_l_Vmm' in row:
                    v_pi_l = row['v_pi_l_Vmm']
                elif 'vpil' in row:
                    v_pi_l = row['vpil']
                elif 'v_pi_l' in row:
                    v_pi_l = row['v_pi_l']
```

**שורה 78**: else - אם כל הפרמטרים נמצאו.

**שורה 79**: הערה - חילוץ loss ו-vpil.

**שורה 80**: הערה - בדיקת שמות עמודות שונים.

**שורה 81**: אתחול alpha (אובדן אופטי ב-dB/cm).

**שורה 82**: אתחול v_pi_l (מכפלת V_π*L ב-V*mm).

**שורות 83-88**: חילוץ alpha:
- בדיקה לפי סדר עדיפות של שמות עמודות
- 'loss_at_v_pi_dB_per_cm' → 'loss_db' → 'alpha'

**שורות 90-95**: חילוץ v_pi_l:
- בדיקה לפי סדר עדיפות של שמות עמודות
- 'v_pi_l_Vmm' → 'vpil' → 'v_pi_l'

#### שורות 97-101: בדיקת ערכים תקינים

```python
                # Skip rows with NaN values
                if alpha is None or v_pi_l is None or np.isnan(alpha) or np.isnan(v_pi_l):
                    print(f"  [WARNING] Missing or NaN values in row {index}. Skipping.")
                    skipped_count += 1
                    continue
```

**שורה 97**: הערה - דילוג על שורות עם NaN.

**שורה 98**: בדיקה:
- אם alpha או v_pi_l הם None
- אם אחד מהם הוא NaN

**שורה 99**: הודעת אזהרה.

**שורה 100**: הגדלת מונה הדילוגים.

**שורה 101**: המשך לשורה הבאה.

#### שורות 103-108: חישוב עלות ורישום

```python
                # Calculate cost
                cost = calculate_loss_function(alpha, v_pi_l)
                
                # Register with optimizer
                optimizer.register(params=params, target=cost)
                registered_count += 1
```

**שורה 103**: הערה - חישוב עלות.

**שורה 104**: קריאה ל-calculate_loss_function לחישוב עלות.

**שורה 106**: הערה - רישום באופטימיזר.

**שורה 107**: רישום נקודת הנתונים עם פרמטרים וערך המטרה (עלות).

**שורה 108**: הגדלת מונה ההצלחות.

#### שורות 110-113: טיפול בשגיאות

```python
        except Exception as e:
            print(f"  [WARNING] Error processing row {index}: {e}. Skipping.")
            skipped_count += 1
            continue
```

**שורה 110**: catch של כל שגיאה.

**שורה 111**: הודעת אזהרה עם פרטי השגיאה.

**שורה 112**: הגדלת מונה הדילוגים.

**שורה 113**: המשך לשורה הבאה.

### שורות 115-119: סיכום והחזרה

```python
    print(f"  -> Registered {registered_count} data points with optimizer")
    if skipped_count > 0:
        print(f"  -> Skipped {skipped_count} rows due to errors or missing data")
    
    return optimizer
```

**שורה 115**: הודעת סיכום - מספר נקודות שנרשמו.

**שורות 116-117**: הודעת אזהרה אם היו דילוגים.

**שורה 119**: החזרת האופטימיזר המאומן.

---

## שורות 122-178: פונקציה `get_next_sample`

### שורות 122-139: חתימת הפונקציה ותיעוד

```python
def get_next_sample(optimizer, result_csv_path=None):
    """
    Predicts next parameter set to sample using Bayesian Optimization.
    
    Args:
        optimizer (BayesianOptimization): Trained BayesianOptimization object
        result_csv_path (str, optional): Path to result.csv file (for reference).
                                        If None, uses config.RESULTS_CSV_FILE
    
    Returns:
        dict: Dictionary containing next parameter values to sample
              Keys match parameter names in config.SWEEP_PARAMETERS
    
    Process:
        1. Uses Gaussian Process model to predict objective function
        2. Uses Acquisition Function (UCB) to select next point
        3. Returns parameter dictionary for next simulation
    """
```

**שורה 122**: הגדרת הפונקציה. מקבלת אופטימיזר מאומן.

**שורות 123-139**: תיעוד:
- תפקיד: חיזוי קבוצת פרמטרים הבאה לבדיקה
- פרמטרים: אופטימיזר מאומן, נתיב CSV (לצורכי הפניה)
- החזרה: מילון עם ערכי הפרמטרים הבאים
- תהליך: שימוש ב-Gaussian Process ו-UCB לבחירת הנקודה הבאה

### שורות 140-141: בדיקת תקינות אופטימיזר

```python
    if optimizer is None:
        raise ValueError("Optimizer is None. Train the optimizer first.")
```

**שורה 140**: בדיקה אם האופטימיזר הוא None.

**שורה 141**: זריקת שגיאה אם האופטימיזר לא הוגדר.

### שורות 143-177: לוגיקת בחירת הנקודה הבאה

```python
    try:
        # Use suggest() to get the next point to sample
        # The acquisition function is already set in the optimizer
        next_point = optimizer.suggest()
        
        if next_point is None:
            print("  [WARNING] Optimizer suggest() returned None")
            return None
        
        # next_point is a dictionary with parameter names as keys
        # Verify all parameters are present and within bounds
        param_names = list(config.SWEEP_PARAMETERS.keys())
        
        for param_name in param_names:
            if param_name not in next_point:
                print(f"  [WARNING] Parameter '{param_name}' missing from suggested point")
                return None
            
            # Clip to bounds if needed (should not be necessary but safety check)
            min_val = config.SWEEP_PARAMETERS[param_name]['min']
            max_val = config.SWEEP_PARAMETERS[param_name]['max']
            
            if next_point[param_name] < min_val:
                print(f"  [WARNING] Clipping {param_name} from {next_point[param_name]} to {min_val}")
                next_point[param_name] = min_val
            elif next_point[param_name] > max_val:
                print(f"  [WARNING] Clipping {param_name} from {next_point[param_name]} to {max_val}")
                next_point[param_name] = max_val
        
        print(f"  -> Next suggested point: {next_point}")
        return next_point
        
    except Exception as e:
        print(f"  [ERROR] Failed to get next sample: {e}")
        return None
```

**שורה 143**: תחילת try-except.

**שורות 144-145**: הערות - שימוש ב-suggest().

**שורה 146**: קריאה ל-suggest() לקבלת הנקודה הבאה.

**שורות 148-150**: בדיקה אם התוצאה היא None:
- הדפסת אזהרה והחזרת None.

**שורות 152-153**: הערות - next_point הוא מילון.

**שורה 154**: רשימת שמות הפרמטרים.

**שורה 156**: לולאה על כל שם פרמטר.

**שורות 157-159**: בדיקה אם הפרמטר קיים ב-next_point:
- אם חסר - הדפסת אזהרה והחזרת None.

**שורות 161-170**: בדיקת גבולות (clipping):
- **שורה 161**: הערה - clipping כבדיקת בטיחות.
- **שורה 162**: קבלת ערך מינימום מה-config.
- **שורה 163**: קבלת ערך מקסימום מה-config.
- **שורות 165-167**: אם הערך קטן ממינימום - clipping למינימום.
- **שורות 168-170**: אם הערך גדול ממקסימום - clipping למקסימום.

**שורה 172**: הדפסת הנקודה המוצעת.

**שורה 173**: החזרת מילון הפרמטרים.

**שורות 175-177**: טיפול בשגיאות:
- הדפסת הודעת שגיאה והחזרת None.

---

## שורות 180-217: פונקציה `calculate_loss_function`

### שורות 180-199: חתימת הפונקציה ותיעוד

```python
def calculate_loss_function(alpha, v_pi_l, weights=None, targets=None):
    """
    Calculates the loss/objective function value from simulation results.
    Returns NEGATIVE value for maximization (BayesianOptimization maximizes by default).
    
    Args:
        alpha (float): Optical loss in dB/cm
        v_pi_l (float): V_π*L product in V*mm (will be converted to V*cm)
        weights (dict, optional): Weights for different metrics.
                                 If None, uses config.FOM_WEIGHTS
        targets (dict, optional): Target values for optimization.
                                 If None, uses config.TARGETS
    
    Returns:
        float: Negative cost value (for maximization)
    
    Formula:
        cost = w_loss * (alpha / target_loss) + w_vpil * (vpil / target_vpil)
        return -cost  (negative for maximization - lower cost = better)
    """
```

**שורה 180**: הגדרת הפונקציה:
- alpha: אובדן אופטי
- v_pi_l: מכפלת V_π*L
- weights: משקלות (אופציונלי)
- targets: ערכי מטרה (אופציונלי)

**שורות 181-199**: תיעוד:
- תפקיד: חישוב פונקציית עלות/מטרה
- חוזר ערך שלילי (כי האופטימיזר ממקסם)
- נוסחה: עלות משוקללת ונורמלית, מוחזר ערך שלילי

### שורות 200-204: שימוש בערכי ברירת מחדל

```python
    # Use default weights and targets from config if not provided
    if weights is None:
        weights = config.FOM_WEIGHTS
    if targets is None:
        targets = config.TARGETS
```

**שורה 200**: הערה - שימוש בערכי ברירת מחדל.

**שורות 201-202**: אם weights לא הועבר, שימוש ב-config.FOM_WEIGHTS.

**שורות 203-204**: אם targets לא הועבר, שימוש ב-config.TARGETS.

### שורות 206-207: המרת יחידות

```python
    # Convert v_pi_l from V*mm to V*cm (divide by 10)
    v_pi_l_cm = v_pi_l / 10.0
```

**שורה 206**: הערה - המרה מ-V*mm ל-V*cm.

**שורה 207**: חלוקה ב-10 להמרת יחידות.

### שורות 209-211: נורמליזציה

```python
    # Normalize by targets
    norm_loss = alpha / targets['loss']
    norm_vpil = v_pi_l_cm / targets['vpil']
```

**שורה 209**: הערה - נורמליזציה לפי ערכי מטרה.

**שורה 210**: נורמליזציה של alpha לפי target_loss.

**שורה 211**: נורמליזציה של v_pi_l לפי target_vpil.

### שורות 213-214: חישוב עלות משוקללת

```python
    # Calculate weighted cost
    total_cost = weights['loss'] * norm_loss + weights['vpil'] * norm_vpil
```

**שורה 213**: הערה - חישוב עלות משוקללת.

**שורה 214**: חישוב: סכום משוקלל של שני המרכיבים המנורמלים.

### שורות 216-217: החזרת ערך שלילי

```python
    # Return negative for maximization (lower cost = higher value)
    return -total_cost
```

**שורה 216**: הערה - החזרת ערך שלילי למקסימיזציה.

**שורה 217**: החזרת הערך השלילי של העלות (עלות נמוכה = ערך גבוה יותר).

---

## שורות 220-269: פונקציה `get_best_result`

### שורות 220-230: חתימת הפונקציה ותיעוד

```python
def get_best_result(result_csv_path=None):
    """
    Returns the best result found so far from results CSV.
    
    Args:
        result_csv_path (str, optional): Path to result.csv file.
                                        If None, uses config.RESULTS_CSV_FILE
    
    Returns:
        dict: Best result row as dictionary, or None if no valid results
    """
```

**שורה 220**: הגדרת הפונקציה. פרמטר אופציונלי - נתיב לקובץ תוצאות.

**שורות 221-230**: תיעוד:
- תפקיד: החזרת התוצאה הטובה ביותר שנמצאה עד כה
- פרמטרים: נתיב לקובץ CSV (אופציונלי)
- החזרה: שורה של התוצאה הטובה ביותר כמילון, או None אם אין תוצאות תקינות

### שורות 231-240: טעינת הקובץ

```python
    if result_csv_path is None:
        result_csv_path = config.RESULTS_CSV_FILE
    
    if not os.path.exists(result_csv_path):
        return None
    
    df = pd.read_csv(result_csv_path)
    
    if len(df) == 0:
        return None
```

**שורות 231-232**: אם לא הועבר נתיב, שימוש בערך מ-config.

**שורות 234-235**: אם הקובץ לא קיים, החזרת None.

**שורה 237**: טעינת הקובץ ל-DataFrame.

**שורות 239-240**: אם הטבלה ריקה, החזרת None.

### שורות 242-244: אתחול משתנים לחיפוש

```python
    # Calculate cost for each row and find the best (lowest cost = highest negative value)
    best_cost = float('inf')
    best_row = None
```

**שורה 242**: הערה - חישוב עלות לכל שורה ומציאת הטובה ביותר.

**שורה 243**: אתחול best_cost לאינסוף (ימצא מינימום).

**שורה 244**: אתחול best_row ל-None.

### שורות 246-267: לולאת חיפוש התוצאה הטובה ביותר

```python
    for index, row in df.iterrows():
        try:
            alpha = None
            v_pi_l = None
            
            if 'loss_at_v_pi_dB_per_cm' in row:
                alpha = row['loss_at_v_pi_dB_per_cm']
            if 'v_pi_l_Vmm' in row:
                v_pi_l = row['v_pi_l_Vmm']
            
            if alpha is None or v_pi_l is None or np.isnan(alpha) or np.isnan(v_pi_l):
                continue
            
            # Calculate cost (positive value, lower is better)
            cost = -calculate_loss_function(alpha, v_pi_l)
            
            if cost < best_cost:
                best_cost = cost
                best_row = row.to_dict()
                
        except Exception:
            continue
```

**שורה 246**: לולאה על כל שורה.

**שורה 247**: תחילת try-except.

**שורות 248-249**: אתחול alpha ו-v_pi_l ל-None.

**שורות 251-254**: חילוץ ערכים:
- **שורות 251-252**: חילוץ alpha (אם קיים).
- **שורות 253-254**: חילוץ v_pi_l (אם קיים).

**שורה 256**: בדיקה אם הערכים תקינים (לא None ולא NaN).

**שורה 257**: דילוג על השורה אם הערכים לא תקינים.

**שורות 259-260**: חישוב עלות:
- קריאה ל-calculate_loss_function
- מינוס כדי לקבל ערך חיובי (עלות נמוכה = טוב יותר)

**שורות 262-264**: עדכון התוצאה הטובה ביותר:
- אם העלות הנוכחית נמוכה יותר, עדכון best_cost ו-best_row.

**שורות 266-267**: אם יש שגיאה, דילוג על השורה.

### שורה 269: החזרת התוצאה

```python
    return best_row
```

החזרת השורה הטובה ביותר כמילון, או None אם לא נמצאה תוצאה תקינה.

---

## סיכום כללי

הקובץ `BO.py` מכיל מודול מלא לאופטימיזציה בייסיאנית של פרמטרים:

1. **train_optimizer**: אימון האופטימיזר עם נתונים קודמים מ-CSV
2. **get_next_sample**: בחירת הנקודה הבאה לבדיקה באמצעות Gaussian Process ו-UCB
3. **calculate_loss_function**: חישוב פונקציית עלות משוקללת מהתוצאות
4. **get_best_result**: מציאת התוצאה הטובה ביותר שנמצאה עד כה

המודול משתמש בספריית `bayes_opt` ומשלב אותה עם מערכת ה-config המקומית לניהול פרמטרים ואופטימיזציה יעילה.
