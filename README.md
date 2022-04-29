אני חושב אולי עשיתי את סעיפים 3-4 בצורה "טובה מדי" ממה שהתכוונו. כאילו, הוא רשם בסוגריים בשאלה להשתמש בnested loops עבור הערכים של הקודקודים, בעוד שאני השתמשתי בproduct. אני חשבתי אולי הוא פשוט המליץ על nested loops. 
שאלות 3-4 מחזירות אותו פלט כמו שאלות 5-6, אז השוויתי בין זמני הריצה וקיבלתי שב3-4 יצא יותר מהר:

    import time
    
    tic_s1 = time.perf_counter()
    for i in range(1000):        # First method 1000 times
        [f'Z(temp={i})  =  {Z_temp(temp=i, ex=4)}\n' for i in [1, 1.5, 2]] 
    tic_e1 = time.perf_counter()

    tic_s2 = time.perf_counter()
    for i in range(1000):        # Second method 1000 times
        [f'Z(temp={i})  =  {Z_temp(temp=i, ex=6)}\n' for i in [1, 1.5, 2]]
    tic_e2 = time.perf_counter()

    print(f' First methods time: {tic_e1 - tic_s1:0.4f} seconds')   -->  First methods time: 25.0807 seconds
    print(f'Second methods time: {tic_e2 - tic_s2:0.4f} seconds')   -->  Second methods time: 97.1103 seconds
    
    
    שאלתי בקבוצה אם זה סבבה...
