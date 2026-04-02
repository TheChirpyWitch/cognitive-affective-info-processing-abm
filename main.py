from hybrid_model import build_demo, make_demo_messages

model = build_demo()

for _ in range(60):
    exo = make_demo_messages(model.t)
    model.step(exo)

model.plot_history()
