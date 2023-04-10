import functions as fn
import graph

if __name__ == "__main__":
    graph.sensitivity_run(
        [fn.bev_holt_model, fn.allee_model, fn.ricker_model, fn.linear_model]
    )

    # Beverton Holt
    graph.ordinary_plot_over_time(fn.bev_holt_model, 20)

    # Allee
    graph.ordinary_plot_over_time(fn.allee_model, 20)

    # Ricker
    graph.ordinary_plot_over_time(fn.ricker_model, 20)

    # Linear
    graph.ordinary_plot_over_time(fn.linear_model, 20)
