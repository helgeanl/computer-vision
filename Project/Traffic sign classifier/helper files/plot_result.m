figure(1);clf(1)
subplot(1,2,1)
plot(loss)
xlabel('Epochs')
title('Loss')
grid on
xlim([0 50])
legend('Training','Validation')
subplot(1,2,2)
plot(acc.*100)
title('Accuracy')
xlabel('Epochs')
grid on
xlim([0 50])