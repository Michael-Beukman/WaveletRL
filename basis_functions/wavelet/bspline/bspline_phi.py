def psi(x: float, order: int) -> float:
    """
    The psi function of the bspline wavelet
    :param t:
    :param order:
    :return:
    """
    if (order == 0):
        if (x > 0 and x <= 0.5): return 1;
        if (x > 0.5 and x <= 1): return -1;
    elif (order == 1):
        def f(t):
            if 0 <= t < 0.5:    return t
            if 0.5 <= t < 1:    return - 7 * t + 4
            if 1 <= t < 1.5:    return 16 * t - 19
            if 1.5 <= t < 2:    return -16 * t + 29
            if 2 <= t < 2.5:    return 7 * t -17
            if 2.5 <= t < 3:    return -t ** 1 + 3
            return 0
        return 1/6 * f(x)
    elif (order == 2):
        if (x > 0 and x <= 0.5): return pow(x, 2) / 240.0;
        if (x > 0.5 and x <= 1): return (-8 + (32 - 31 * x) * x) / 240.0;
        if (x > 1 and x <= 1.5): return (229 - 442 * x + 206 * pow(x, 2)) / (240.0);
        if (x > 1.5 and x <= 2): return (-1643 + 2054 * x - 626 * pow(x, 2)) / (240.0);
        if (x > 2 and x <= 2.5): return (-5 + 2 * x) * (-339 + 176 * x) / (80.0);
        if (x > 2.5 and x <= 3): return (-5 + 2 * x) * (-541 + 176 * x) / (-80.0);
        if (x > 3 and x <= 3.5): return (7023 - 4206 * x + 626 * pow(x, 2)) / 240.0;
        if (x > 3.5 and x <= 4): return (-3169 + 2 * (809 - 103 * x) * x) / (240.0);
        if (x > 4 and x <= 4.5): return (623 + x * (-278 + 31 * x)) / (240.0);
        if (x > 4.5 and x <= 5): return pow(x - 5, 2) / (-240.0);
    elif (order == 3):
        # hacky maybe
        def f(t):
            x = t
            if 0 <= t < 0.5:
                return t ** 2
            elif 0.5 <= t < 1:
                return -8 + 32 * t - 31 * t ** 2
            elif 1 <= t < 1.5:
                return 229 - 442 * t + 206 * t ** 2
            elif 1.5 <= t < 2:
                return -1643 + 2054 * t - 626 * t ** 2
            elif 2 <= t < 2.5:
                return 4 * (1695 - 1558 * t + 352 * t ** 2)
            elif 2.5 <= t < 3:
                return 4 * (-2705 + 1962 * t - 352 * t ** 2)
            elif 3 <= t < 3.5:
                return 7023 - 4206 * t + 626 * t ** 2
            elif 3.5 <= t < 4:
                return -3169 + 1618 * t - 206 * t ** 2
            elif 4 <= t < 4.5:
                return 623 - 278 * t + 31 * t ** 2
            elif 4.5 <= t < 5:
                return -(-5 + t) ** 2
            else:
                return 0
        return 1 / 240 * f(x)
    elif (order == 4):
        return 0;
    return 0;

def phi(x: float, order:int):
    if (order == 0):
        if (x > 0 and x <= 1): return 1;
    elif (order == 1):
        if (x > 0 and x <= 1): return x;
        if (x > 1 and x <= 2): return 2 - x;
    elif (order == 2):
        if (x > 0 and x <= 1): return (0.5 * pow(x, 2))
        if (x > 1 and x <= 2): return (0.75 - pow((x - 1.5), 2))
        if (x > 2 and x <= 3): return (0.5 * pow(x - 3, 2))
    elif (order == 3):
        if (x > 0 and x <= 1): return (pow(x, 3) / 6.0)
        if (x > 1 and x <= 2): return (2 / 3.0 - x * pow(x - 2, 2) / 2)
        if (x > 2 and x <= 3): return (-22 / 3.0 + 0.5 * x * (20 + x * (x - 8)))
        if (x > 3 and x <= 4): return (-pow(x - 4, 3) / 6.0)
    elif (order == 4):
        if (x > 0 and x <= 1): return pow(x, 4) / 24.0;
        if (x > 1 and x <= 2): return (-5 + 2 * x * (10 + x * (-15 - 2 * x * (x - 5)))) / 24.0;
        if (x > 2 and x <= 3): return (155 + 6 * x * (x - 5) * (10 + x * (x - 5))) / 24.0;
        if (x > 3 and x <= 4): return (-655 + 2 * x * (390 + x * (-165 - 2 * x * (x - 15)))) / 24.0;
        if (x > 4 and x <= 5): return pow(x - 5, 4) / 24.0;

    return 0;


def phi_normal(x: float, order: int):
        if (order == 0) :
            if (x>0 and x<=1): return 1;
        elif (order == 1) :
            if (x>0 and x<= 1): return 1.22474*x;
            if (x>1 and x<= 2): return 1.22474*(2-x);
        elif (order == 2) :
            if (x>0 and x<= 1): return 1.3484*(0.5*pow(x, 2));
            if (x>1 and x<= 2): return 1.3484*(0.75 - pow((x-1.5),2));
            if (x>2 and x<= 3): return 1.3484*(0.5*pow(x-3, 2));
        elif (order == 3) :
            if (x>0 and x<= 1): return 1.44433*(pow(x, 3)/6.0);
            if (x>1 and x<= 2): return 1.44433*(2/3.0 -x*pow(x-2,2)/2);
            if (x>2 and x<= 3): return 1.44433*(-22/3.0+0.5*x*(20+x*(x-8)));
            if (x>3 and x<= 4): return 1.44433*(-pow(x-4, 3)/6.0);
        elif (order == 4) :
            if (x>0 and x<= 1): return 1.52425*pow(x, 4)/24.0;
            if (x>1 and x<= 2): return 1.52425*(-5+2*x*(10+x*(-15-2*x*(x-5))))/24.0;
            if (x>2 and x<= 3): return 1.52425*(155+6*x*(x-5)*(10+x*(x-5)))/24.0;
            if (x>3 and x<= 4): return 1.52425*(-655+2*x*(390+x*(-165-2*x*(x-15))))/24.0;
            if (x>4 and x<= 5): return 1.52425*pow(x-5, 4)/24.0;
        
        return 0;
