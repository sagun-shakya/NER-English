import torch.nn.functional as F
import utils

class DynamicModel:
    def __init__(self, model, optimizer, tag_pad_id):
        self.model = model
        self.optimizer = optimizer
        self.tag_pad_id = tag_pad_id

    def fit(self, train_loader, test_loader, epochs, n = 50, early_stopping_callback = None, return_cache = True, plot_history = True):
        ######################## INITIALIZATION CACHE ########################
        # Cache for accuracy and losses in each epoch for training and validatin sets.
        accuracy_cache_train = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}
        accuracy_cache_val = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}

        loss_cache_train = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}
        loss_cache_val = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}

        # Fine-grained accuracy and loss cache to see how these values evolve for each batch within each epoch.
        accuracy_cache_train_grained = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}
        accuracy_cache_val_grained = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}

        loss_cache_train_grained = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}
        loss_cache_val_grained = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}

        ######################## TRAINING PHASE ########################
        for ee in range(epochs):    
            
            # Empty lists for storing the train/validation accuracy and losses for each n-th iteration in the ee-th epoch.
            # These will be stored in the dictionary containing the cache for each epoch.    
            accuracy_train = []
            accuracy_val = []
            
            loss_train = []
            loss_val = []
            
            running_loss = 0
            running_accuracy = 0
            for ii, ((sample, seq_len), tag) in enumerate(train_loader):
                # Clear the gradients.
                self.model.zero_grad()

                # Generating the output of shape (batch_size, num_tags, max_len).
                preds = self.model.forward(sample, seq_len)
                if preds is None:
                    print("Skipped in batch: ", ii)
                    continue        

                # Using negative log-likelihood loss function. 
                # If CrossEntropyLoss is used, there's no need to apply softmax. Can take input directly from the last linear layer.
                # Preventing the <pad> element from contributing to the loss.
                loss = F.nll_loss(preds, tag, ignore_index = self.tag_pad_id)
                loss_rounded = round(loss.item(), 4)

                running_loss += loss_rounded
                
                #print(f"\nLoss value in this iteration (Epoch : {ee} & Batch : {ii}): {loss_rounded}")    # For debugging.

                # Backpropagation.
                loss.backward()

                # Update the weights.
                self.optimizer.step()

                # Categorical Accuracy.
                train_acc_per_iter = utils.categorical_accuracy(preds, tag, tag_pad_value = self.tag_pad_id)
                running_accuracy += train_acc_per_iter
                #print(f"Accuracy in this iteration (Epoch : {ee} & Batch : {ii}): {train_acc_per_iter}")    # For debugging.

                # Calculate the loss and accuracy for the validation set in every 50 iteration.
                if (ii + 1) % n == 0:
                    avg_train_accuracy = running_accuracy / n
                    avg_train_loss = running_loss / n

                    # Setting the running loss and accuracy to 0 to be able to collect these values in 51st, 101st, ... iteration.
                    running_loss = 0
                    running_accuracy = 0

                    accuracy_train.append(avg_train_accuracy)
                    loss_train.append(avg_train_loss)

                    avg_val_accuracy = 0
                    avg_val_loss = 0
                    
                    for jj, ((sample_t, seq_len_t), tag_t) in enumerate(test_loader, 1):               
                        # Forward Propagation.
                        preds_val = self.model.forward(sample_t, seq_len_t)
                        
                        # Calculate the loss.
                        val_loss_per_batch = F.nll_loss(preds_val, tag_t, ignore_index = self.tag_pad_id)
                        val_loss_per_batch = val_loss_per_batch.item()
                        
                        # Calculating the accuracy.
                        val_accuracy_per_batch = utils.categorical_accuracy(preds_val, tag_t, tag_pad_value = self.tag_pad_id)
                        
                        # Storing the losses and accuracies for validation batches (NOT THE TRAINING BATCH) in this iteration.
                        ## Validation Set.
                        avg_val_accuracy += val_accuracy_per_batch
                        avg_val_loss += val_loss_per_batch
                    
                    # Calculating the average loss for the valdation set (all of the batches) in this iteration.
                    # jj amounts to len(test_loader) at the end of validation loop.
                    avg_val_accuracy = avg_val_accuracy / jj
                    avg_val_loss = avg_val_loss / jj

                    # Storing the above accuracy and loss for 50th, 100th, ... iterations in the training loop.
                    accuracy_val.append(avg_val_accuracy)
                    loss_val.append(avg_val_loss)
                    
                    # Verbose.
                    epoch_step_info = f"Epoch [{ee+1} / {epochs}], Step [{ii+1} / {len(train_loader)}], "
                    loss_info = f"Training Loss: {avg_train_loss:.4f}, Validation loss: {avg_val_loss:.4f}, "
                    accuracy_info = f"Training Accuracy: {avg_train_accuracy:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}"
                    print(epoch_step_info + loss_info + accuracy_info + '\n')
                    
                if ii == 200:
                    break
                
            # Storing the cache for this epoch into the dictionaries above.
            
            # Train Set.
            accuracy_cache_train['epoch_' + str(ee + 1)] = utils.compute_average(accuracy_train)
            loss_cache_train['epoch_' + str(ee + 1)] = utils.compute_average(loss_train)

            accuracy_cache_train_grained['epoch_' + str(ee + 1)] = accuracy_train
            loss_cache_train_grained['epoch_' + str(ee + 1)] = loss_train
            
            # Validation Set.
            accuracy_cache_val['epoch_' + str(ee + 1)] = utils.compute_average(accuracy_val)
            loss_cache_val['epoch_' + str(ee + 1)] = utils.compute_average(loss_val)
            
            accuracy_cache_val_grained['epoch_' + str(ee + 1)] = accuracy_val
            loss_cache_val_grained['epoch_' + str(ee + 1)] = loss_val

            if early_stopping_callback is not None:
                early_stopping_callback(loss_cache_val['epoch_' + str(ee + 1)], self.model)
                if early_stopping_callback.early_stop:
                    print("Early Stopping in Epoch ", ee)
                    break

        if plot_history:
            training_loss_per_epoch_list = [*loss_cache_train.values()]
            val_loss_per_epoch_list = [*loss_cache_val.values()]

            training_accuracy_per_epoch_list = [*accuracy_cache_train]
            val_accuracy_per_epoch_list = [*accuracy_cache_val]

            ax = utils.plot_history_object(
                training_loss_per_epoch_list, 
                val_loss_per_epoch_list,
                training_accuracy_per_epoch_list,
                val_accuracy_per_epoch_list,
                epochs,
                figsize = (15,7), 
                style = 'dark_background')

        if return_cache:            
            history = {
            'accuracy_train' : accuracy_cache_train,
            'accuracy_val' : accuracy_cache_val,
            'loss_train' : loss_cache_train,
            'loss_val' : loss_cache_val,
            'accuracy_train_grained' : accuracy_cache_train_grained,
            'accuracy_val_grained' : accuracy_cache_val_grained,
            'loss_train_grained' : loss_cache_train_grained,
            'loss_val_grained' : loss_cache_val_grained
            }

            return history
            
            
