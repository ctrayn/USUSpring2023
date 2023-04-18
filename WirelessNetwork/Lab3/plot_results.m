function plot_results(results1, results2, x_axis, the_legend, the_title)
    figure();
    hold on
    plot(x_axis, results1)
    plot(x_axis, results2)
    title(the_title)
    legend(the_legend, 'Location', 'southeast')
    xlabel("SINR[db]")
    ylabel("Converge probability")
end