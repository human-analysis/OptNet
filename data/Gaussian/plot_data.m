function  plot_data(X, Y, S, n)

% scatter(X(1,1:n/4),X(2,1:n/4),'bX','LineWidth',2)
% 
% hold on
% scatter(X(1,n/4+1:n/2),X(2,n/4+1:n/2),'rX','LineWidth',2)
% 
% hold on
% scatter(X(1,n/2+1:3*n/4),X(2,n/2+1:3*n/4),'filled','Ob','LineWidth',2)
% 
% hold on
% scatter(X(1,3*n/4+1:n),X(2,3*n/4+1:n),'filled','Or','LineWidth',2)

redX = find(S==0&Y==0);
redO = find(S==0&Y==1);
blueX = find(S==1&Y==0);
blueO = find(S==1&Y==1);

scatter(X(1, redX), X(2, redX), 'rX', 'LineWidth',2)

hold on
scatter(X(1, redO), X(2, redO),'filled', 'rO','LineWidth',2)

hold on
scatter(X(1, blueX), X(2, blueX), 'bX','LineWidth',2)

hold on
scatter(X(1, blueO), X(2, blueO),'filled', 'bO','LineWidth',2)
% axis([-4, 4, -4, 4])
% xticks([-2, -1, 0, 1, 2])
% yticks([-2, -1, 0, 1, 2])

grid
set(gca, 'FontSize', 14)

box on
end
