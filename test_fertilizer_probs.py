from Agro import AgroAidBot

bot = AgroAidBot()

class DummyGUI:
    def show_graph(self, data, title, position='center'):
        print(f"\n--- {title} ({position}) ---")
        for k, v in data.items():
            print(f"{k}: {v}")
        # also print a summary line
        try:
            top = max(data.items(), key=lambda i: i[1])
            print(f"Top: {top[0]} ({top[1]}{('%' if '%' in title or 'Percent' in title else '')})")
        except Exception:
            pass

bot.gui = DummyGUI()

bot.inputs = {
    'Temperature':'74',
    'Humidity':'52',
    'Moisture':'54',
    'Soil_Type':'Sandy',
    'Crop':'Sugarcane',
    'Nitrogen':'56',
    'Potassium':'54',
    'Phosphorus':'54'
}

bot.run_fertilizer_prediction()
print('\nTest complete')