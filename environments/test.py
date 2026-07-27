import numpy as np

cLeaf = 9.5283e4    # cLeaf
cStem = 2.5107e5    # cStem
cFruit = 5.5338e4    # cFruit
print(sum([cLeaf, cStem, cFruit]))

def plant_state_for_day(day):
    """
    Calculate the plant weight and canopy temperature sum for a given day of the year.
    
    At day_start (59) first of March,
        the plant weight is half of the final weight and the canopy temperature sum is half of the final value.
    At day_end (181) first of July,
        the plant weight reaches the final weight and the canopy temperature sum reaches the final value.
    
    Parameters:
    -----------
    day : float or int
        The current day of the year
    
    Returns:
    --------
    float, float
        The plant weight at the given day and the canopy temperature sum at the given day
    """

    w_final=401691.0
    tCanSum_final=3.0978e3
    day_start=60
    day_end=182
    # Initial weight is half of final weight
    w_initial = w_final / 2.0
    tCanSum_initial = tCanSum_final / 2.0
    
    # Normalize day to [0, 1] range
    t = (day - day_start) / (day_end - day_start)
        
    # Sinusoidal growth (smooth S-curve)
    # Using sine function: sin(t * π/2) goes from 0 to 1 as t goes from 0 to 1
    # This gives a smooth acceleration curve
    sine_scaled = np.sin(t * np.pi / 2)
    weight = w_initial + (w_final - w_initial) * sine_scaled
    tCanSum = tCanSum_initial + (tCanSum_final - tCanSum_initial) * sine_scaled
    return weight, tCanSum

if __name__ == "__main__":
    cWeight = cLeaf + cStem + cFruit
    print(f"Final plant weight (cWeight): {cWeight:.2f} g")
    print(f"Initial plant weight (50% of final): {cWeight/2:.2f} g")
    print("\n" + "=" * 60)

    days = [59, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 181]

    for day in days:
        weight, tCanSum = plant_state_for_day(day)
        ratio = weight / cWeight
        tCanSum_ratio = tCanSum / (3.0978e3)
        print(f"Day: {day:3d}: {weight:12.2f} g ({ratio:6.1%} of final) (should be {weight/cWeight:.1%})")
        print(f"Day: {day:3d}: {tCanSum:12.2f} ({tCanSum_ratio:.1%} of final)")
    