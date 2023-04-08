%% Problem 8.6

syms m mu xmm1 xm xmp1 xmp2 T c3 c2 c1 c0

A = [
    ((m-1)*T)^3 ((m-1)*T)^2 ((m-1)*T) 1;
    (m*T)^3     (m*T)^2     (m*T)     1;
    ((m+1)*T)^3 ((m+1)*T)^2 ((m+1)*T) 1;
    ((m+2)*T)^3 ((m+2)*T)^2 ((m+2)*T) 1;
    ];

b = [xmm1; xm; xmp1; xmp2];

cs = inv(A)*b;
disp(cs)