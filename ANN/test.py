
diff = []    

l = 2

for j in range(11):
    for i in np.arange(1e3):
        idx = np.random.randint(0, ann.n_train, ann.batch_size)
        
        #
        phi = 10**-(1+j)
        ann.feed_forward(X[idx], batch_size = ann.batch_size)
        dydX = ann.jacobian(X[idx], batch_size = ann.batch_size)
        dydW = ann.layers[l].y_grad_W
        X_hat = X[idx] + 2.0*phi*dydX.T
        ann.feed_forward(X_hat, batch_size = ann.batch_size)
        dydX2 = ann.jacobian(X_hat, batch_size = ann.batch_size)
        #dydX_hat = ann.jacobian(X_hat, batch_size = ann.batch_size)
        dydW_hat = ann.layers[l].y_grad_W
        d2y_dWdX = (dydW_hat - dydW)/phi
        
        #
        phi = 10**-(2+j)
        ann.feed_forward(X[idx], batch_size = ann.batch_size)
        dydX_test = ann.jacobian(X[idx], batch_size = ann.batch_size)
        dydW_test = ann.layers[l].y_grad_W
        X_hat_test = X[idx] + 2.0*phi*dydX_test.T
        ann.feed_forward(X_hat_test, batch_size = ann.batch_size)
        dydX2_test = ann.jacobian(X_hat_test, batch_size = ann.batch_size)
        dydW_hat_test = ann.layers[l].y_grad_W
        d2y_dWdX_test = (dydW_hat_test - dydW_test)/phi
        
        diff.append(np.linalg.norm(d2y_dWdX_test - d2y_dWdX, ord=np.inf)/np.linalg.norm(d2y_dWdX_test, ord=np.inf))
    
    plt.plot(diff)

plt.yscale('log')