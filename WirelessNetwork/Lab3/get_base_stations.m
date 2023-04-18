function base_stations = get_base_stations(radius, reuse)
    if reuse == 1/3
        base_stations = [
            [0 0]
            [ radius*3 0]
            [-radius*3 0]
            [ radius*3*cos(60)  radius*3*sin(60)]
            [ radius*3*cos(120) radius*3*sin(120)]
            [ radius*3*cos(240) radius*3*sin(240)]
            [ radius*3*cos(300) radius*3*sin(300)]
        ];
    else
        base_stations = [
            [0 0]
            [ 1.5*radius  0.5*sqrt(3)*radius]
            [ 1.5*radius -0.5*sqrt(3)*radius]
            [-1.5*radius  0.5*sqrt(3)*radius]
            [-1.5*radius -0.5*sqrt(3)*radius]
            [0  2*radius/sqrt(3)]
            [0 -2*radius/sqrt(3)]
        ];
    end
end