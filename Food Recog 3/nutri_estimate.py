import pandas as pd

df = pd.read_excel(r"Food Recog Real Final\dataset\energy&nutrients.xlsx")

df.columns = df.columns.str.strip()

df = df.astype(str)
label   = df.iloc[:, 0].tolist()
unit    = df.iloc[:, 1].tolist()
calo    = df.iloc[:, 2].tolist()
protein = df.iloc[:, 3].tolist()
fiber   = df.iloc[:, 4].tolist()
carb    = df.iloc[:, 5].tolist()
fat     = df.iloc[:, 6].tolist()
nacl    = df.iloc[:, 7].tolist()
nutri   = df.iloc[:, 8].tolist()
note    = df.iloc[:, 9].tolist()


def getDataByLabel(label_value):
    row = df[df.iloc[:, 0] == label_value]
    if row.empty:
        return " coffee "

    data = row.iloc[0].tolist()

    fields = [
        "Unit", "Năng lượng - Calories", "Chất đạm - Protein", "Chất xơ - Fiber",
          "Chất bột đường - Carb", "Chất béo - Fat", "Muối - NaCl", "Các dưỡng chất khác"
    ]

    result_string = f"Mỗi {data[1]} {data[0]} sẽ gồm có: \n"
    for i in range(2, 9):
        field_name = fields[i - 1]
        value = data[i]
        result_string += f"-{field_name}: {value}"
        if field_name == "Năng lượng - Calories": result_string += "kcal\n"
        elif field_name == "Các dưỡng chất khác": result_string += "\n"
        else : result_string += "g\n"

    return result_string

def getValue(label_value, col):
    row = df[df.iloc[:, 0] == label_value]
    if row.empty:
        return None
    return row[col].values[0]
