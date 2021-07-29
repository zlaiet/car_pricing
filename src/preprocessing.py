import pandas as pd
from datetime import date


def get_models(marque):
    model = marque.rsplit(" ", len(marque))[1 : len(marque)]
    m_ = ""
    for mod in model:
        m_ = m_ + mod + " "
    return m_


def models_rand(rand):
    if ("ROVER" or "LAND") in rand:
        model = rand.rsplit(" ", len(rand))[1 : len(rand)]
        m_ = ""
        for mod in model:
            m_ = m_ + mod + " "
        return m_
    else:
        return rand


def rand(name):
    if name == "LAND" or name == "ROVER":
        name = "LANDROVER"
    return name


def permutation(price):
    return float(price)


class clean_data:
    # l=["BERLINE","XENON "," CUPRA","BMW SÉRIE 3 ","MEGANE ","206 ","ASTRA C","QASHQAI+2","COOPERCOOPER","COOPERCOUNTRYMAN","MERCEDES-BENZ GLC ","MERCEDES-BENZ CLASSE C ","MERCEDES-BENZ CLASSE E ","MAZDA 3 ","MAZDA 2 ","LAND ROVER DISCOVERY ","KIA RIO","D-MAX ","HYUNDAI GRAND I10 ","FOCUS "," ABARTH"," CROSSBACK"," CABRIO"," VAN","CITROËN C4 "," CACTUS"," PICASSO","BMW SÉRIE 5 ","BMW SÉRIE 4 "," CABRIOLET","CABRIOLET"," BACK"," GRAN ","AUDI A3 ","BMW SÉRIE 1 ","AUDI A8 L","5 PORTES","COUPé","DC","3 PORTES","EVOQUE","VELAR","SPORT","GT","UTILITAIRE","SEDAN","5P","3P","SPORTS","SPORTBACK","PICK-UP DC","PICK-UP","MULTISPACE","GTI"]
    def del_coupe(file):
        f = file
        if "plus de " in file:
            f = file.replace("plus de ", "")
        return f

    def changeprice(price):
        p = price
        if "DT" in price:
            p = price.replace("DT", "")
        return p

    def kilm(km):
        k = km
        if "km" in km:
            k = km.replace("km", "")
        return k

    def change(price):
        if " " in price:
            price = price.replace(" ", "")
        return price

    def cv(cv):
        k = cv
        if " CV" in cv:
            k = cv.replace(" CV", "")
        return k

    def lower_name(name):
        name = name.lower()
        return name

    def get_date(date_):
        todays_date = date.today()
        year = todays_date.year
        date_ = str(date_)
        date_ = date_.split("-")[1]
        date_ = int(date_)
        age = year - date_
        return age

    def del_tran_speciaux_carac(mot):
        if mot == "intã©grale":
            mot = "integrale"
            return mot
        else:
            return mot

    def del_carr_speciaux_carac(mot):
        if mot == "CoupÃ©" or mot == "coupã©":
            mot = "Coupe"
            return mot
        else:
            return mot

    def clean(data):
        data["marque"] = data["marque"].apply(clean_data.lower_name)
        data["energie"] = data["energie"].apply(clean_data.lower_name)
        data["boite"] = data["boite"].apply(clean_data.lower_name)
        data["transmission"] = data["transmission"].apply(clean_data.lower_name)
        data["transmission"] = data["transmission"].apply(
            clean_data.del_tran_speciaux_carac
        )

        data["couleur"] = data["couleur"].apply(clean_data.lower_name)
        data["carrosserie"] = data["carrosserie"].apply(clean_data.lower_name)
        data["carrosserie"] = data["carrosserie"].apply(
            clean_data.del_carr_speciaux_carac
        )

        data["kilometrage"] = data["kilometrage"].apply(clean_data.kilm)
        data["kilometrage"] = data["kilometrage"].astype(int)

        data["puissance"] = data["puissance"].apply(clean_data.cv)
        data["puissance"] = data["puissance"].apply(clean_data.del_coupe)
        data["puissance"] = data["puissance"].astype(int)

        data["price"] = data["price"].apply(clean_data.changeprice)
        data["price"] = data["price"].apply(clean_data.change)
        data["price"] = data["price"].astype(int)

        data["age"] = data["date_circulation"].apply(clean_data.get_date)

        del data["date_circulation"]
        del data["date_annonce"]
        return data


def main():
    data_ = pd.read_csv(
        "data/cleanData.csv",
        names=[
            "marque",
            "kilometrage",
            "date_circulation",
            "date_annonce",
            "carrosserie",
            "energie",
            "puissance",
            "boite",
            "transmission",
            "couleur",
            "price",
        ],
        encoding="latin1",
        error_bad_lines=False,
    )
    data_["model"] = data_["marque"].apply(get_models)

    data = clean_data.clean(data_)

    data["brand"] = data.marque.str.split(" ").str.get(0).str.upper()

    data["brand"] = data["brand"].apply(rand)

    data["model"] = data["model"].apply(models_rand)

    # del data["brand"]
    del data["marque"]
    data["price_1"] = data["price"].apply(permutation)
    del data["price"]
    data["price"] = data["price_1"].apply(permutation)
    del data["price_1"]
    print(data)


# data.to_csv("data_after_preprocessing.csv", header=None, index=False)

if __name__ == "__main__":
    main()
