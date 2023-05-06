= Player("demo", exploration_rho=0, lr_alpha=0)
    demo_p1.loadPolicy("trained_controller")
    stDemo = State(demo_p1)
    stDemo.play()