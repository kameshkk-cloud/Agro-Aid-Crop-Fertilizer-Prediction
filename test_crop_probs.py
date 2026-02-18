from Agro import AgroAidBot

bot = AgroAidBot()

class DummyGUI:
    def show_graph(self, data, title, position='center'):
        # position argument supported for compatibility with the GUI
        print('\n---', title, '---')
        for k, v in data.items():
            print(f'{k}: {v}')

bot.gui = DummyGUI()

# Provide reasonable sample inputs
bot.inputs = {
    'Nitrogen': '50',
    'Phosphorus': '30',
    'Potassium': '40',
    'Temperature': '25',
    'Humidity': '60',
    'pH_Value': '6.5',
    'Rainfall': '100'
}

bot.run_crop_prediction()
print('\nTest complete')