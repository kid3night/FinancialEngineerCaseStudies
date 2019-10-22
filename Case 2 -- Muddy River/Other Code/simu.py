def new_path(eval_para_pos, n = 5000):
#     N_month = 97
    N = 24
    N *= 4
    dt = 1/48
    thetap1 = [39.28, 40.65, 40.48, 42.75, 46.62, 71.88, 86.92, 103.63, 60.78, 35.55, 39.67, 44.47]
    thetag1 = [2.51, 2.49, 2.66, 2.59, 5.28, 6.33, 6.39, 5.93, 3.12, 2.73, 2.02, 3.07]
    p_theta_std = np.array([1.45, 1.92, 2.36, 2.65, 5.01, 8.66, 12.21, 13.79, 12.78, 14.01, 12.77, 10.9])
    g_theta_std = np.array([0.17, 0.2, 0.28, 0.4, 0.43, 0.67, 0.48, 0.53, 0.45, 0.78, 0.84, 0.71])
    variable_list = [7, 20, 3, 75, 0.083, 35, 0, 3, 75, 7.22, 0.3, 28.87, 10.83, np.array(thetap1), np.array(thetag1)]
    variables_names = ['alphaCC', 'alphaJC', 'alphaG', 
                       'm', 'p', 'CC0', 'JC0', 'G0',
                       'spike_thres', 'sigmaG', 'rho', 
                       'v_summer', 'v_winter', 'PowerTheta', 'GasTheta']
    vec_res_p = list()
    vec_res_g = list()
    month_list = [i for i in range(1, 98)]
    num_ses = np.arange(-3, 3.2, 0.2)
    len_muls = len(num_ses)
    result_ = list()
    
    for num_se in num_ses:

        if eval_para_pos == 0:
            new_variable = 7 - num_se * 0.5
        if eval_para_pos == 1:
            new_variable = 20 - num_se * 1.3
        if eval_para_pos == 2:
            new_variable = 3 - num_se * 0.2


            
        if eval_para_pos == 3:
            new_variable = 75 - num_se * 5
        elif eval_para_pos == 4:
            new_variable = 0.083 - num_se * 0.005
        elif eval_para_pos == 9:
            new_variable = 7.22 - num_se * 0.05
        elif eval_para_pos == 10:
            new_variable = 0.3 - num_se * 0.05
        elif eval_para_pos == 11:
            new_variable = 28.87 - num_se * 2
        elif eval_para_pos == 12:
            new_variable = 10.83 - num_se * 2
        elif eval_para_pos == 13:
            new_variable = variable_list[13] - num_se * p_theta_std 
        elif eval_para_pos == 14:
            new_variable = variable_list[14] - num_se * g_theta_std
        else:
            new_variable = variable_list[eval_para_pos]

        variable_list[eval_para_pos] = new_variable
        alphaCC  = variable_list[0]
        alphaJC  = variable_list[1]
        alphaG  = variable_list[2]
        m  = variable_list[3]
        p  = variable_list[4]
        CC0  = variable_list[5]
        JC0  = variable_list[6]
        G0  = variable_list[7]
        spike_thres  = variable_list[8]
        sigmaG  = variable_list[9]/np.sqrt(dt)/100
        rho  = variable_list[10]
        v_summer  = variable_list[11]/np.sqrt(dt)/100
        v_winter  = variable_list[12]/np.sqrt(dt)/100
        PowerTheta  = variable_list[13]
        GasTheta  = variable_list[14]

        V = np.zeros((N+1,n))
        W = norm.rvs(size = (N+1,n))*np.sqrt(dt)
        Wtilde = norm.rvs(size = (N+1,n))*np.sqrt(dt)
        B = rho*W + np.sqrt(1-rho**2)*Wtilde

        CC = np.zeros((N+1,n)) 
        CC[0,:] = CC0
        JC = np.zeros((N+1,n))
        JC[0,:] = JC0
        G = np.zeros((N+1,n)) 
        G[0,:] = G0
        PC = np.zeros((N+1,n))
        PC[0,:] = CC[0,:]
        Power_MSE,Gas_MSE = 0, 0
        Power_Price_Fit, Gas_Price_Fit, CC_Price_Fit, JC_Price_Fit = list(), list(), list(), list()

        for i in range(1, N + 1):
            month1 = month_list[i] % 12    
            monthIndicator = (month1 > 3)&(month1 < 8)
            V[i,:] = monthIndicator*v_summer + (1 - monthIndicator)*v_winter
            CC[i,:] = alphaCC*(thetap1[month1-1] - CC[i-1,:])*dt + V[i,:]*CC[i-1,:]*W[i,:] + CC[i-1,:]
            JC[i,:] = alphaJC*( 0 - JC[i-1,:])*dt + m*(uniform.rvs() < p)+JC[i-1,:]
            #Power Price
            PC[i,:] = CC[i,:] + JC[i,:]*(PC[i-1,:] > spike_thres)
            #Gas Price
            G[i,:]  = alphaG*(thetag1[month1-1] - G[i-1,:])*dt + sigmaG * G[i-1,:] * B[i,:] + G[i-1,:]
        result_.append(valuation(PC.T, G.T, upper_, lower_))
    results = np.array(result_) / result_[int(len_muls / 2)]
    
    return (variables_names[eval_para_pos], np.mean(np.abs(results - 1)), result_[int(len_muls / 2)])
#     plt.plot(num_ses, results, 'o')
#     plt.title('Final Profit Sensitivity Analysis - {}'.format(variables_names[eval_para_pos]))
#     plt.xlabel('multiplier of SE')
#     plt.ylabel('Profit')
    
#     return PC, G