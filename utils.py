def monthToNum(shortMonth):
    return {
            'jan': 1,
            'feb': 2,
            'mar': 3,
            'apr': 4,
            'may': 5,
            'jun': 6,
            'jul': 7,
            'aug': 8,
            'sep': 9, 
            'oct': 10,
            'nov': 11,
            'dec': 12
    }[shortMonth]

def dayToNum(day):
        return {
            'mon': 0,
            'tue': 0,
            'wed': 0,
            'thu': 0,
            'fri': 1,
            'sat': 1,
            'sun': 1,
    }[day]