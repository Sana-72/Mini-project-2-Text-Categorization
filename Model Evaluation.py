score = model.evaluate(x_test, y_test,batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


445/445 [==============================] - 0s 82us/sample - loss: 0.1187 - acc: 0.9618
Test loss: 0.11872842385527793
Test accuracy: 0.9617978
