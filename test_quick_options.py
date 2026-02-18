from Agro import AgroAidGUI

g = AgroAidGUI()
# simulate fertilizer flow
g.bot.process_input('2')
g.bot.process_input('45')
g.bot.process_input('25')
g.bot.process_input('65')
print('quick_options_visible:', g.quick_options_visible)
# print the children widget names for quick_select_frame for debugging
print('quick_select children:', [type(c).__name__ for c in g.quick_select_frame.winfo_children()])
# cleanup
g.root.destroy()
print('done')