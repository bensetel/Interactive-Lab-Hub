
                    """
                    if 'shoulder' in pose_dict[i]:
                        print('-'*50)
                        xchange = cur_pos.x - init_positions[i].x
                        ychange = cur_pos.y - init_positions[i].y
                        zchange = cur_pos.z - init_positions[i].z
                        print('change for: ', pose_dict[i], 'x : ', xchange)
                        print('change for: ', pose_dict[i], 'y : ', ychange)
                        print('change for: ', pose_dict[i], 'z : ', zchange)
                        print('-'*50)
                        fn = f'{pose_dict[i]}_changes.csv'
                        if not(os.path.isfile(fn)):
                            f = open(fn, 'w+')
                            f.write('x,y,z\n')
                            f.close()
                            
                        f = open(fn, 'a')
                        s = str(xchange) + ',' + str(ychange) + ',' + str(zchange) + '\n'
                        f.write(s)
                        f.close()

                        xangle = pos_to_angle(cur_pos.x, 'x', width, 'left_shoulder')
                        yangle = pos_to_angle(cur_pos.x, 'y', height, 'left_shoulder')
                        print('#'*50)
                        print('xangle:', xangle)
                        print('yangle:', yangle)
                        print('#'*50)
                        #if x, coeff = width
                        #if y, coeff = height
                    """  
                  

                    
                    """
                    print('-'*50)
                    fi = open(f'{pose_dict[i]}.txt', 'a+')
                    fi.write('\n')
                    fi.write(repr(pl[0][i]))
                    fi.close()
                    print('wrote', pose_dict[i])
                    #print(f'pl {pose_dict[i]}', pl[0][i])
                    print('-'*50)
                    change = pl[0][i].y - init_positions[i].y

                    print('change:', change)
                    if (np.abs(change) < threshold[j]):
                        print('threshold not met')
                    else:
                        print('met!')
                        if change > 0:
                            voter[j] += 1
                        else:
                            voter[j] -= 1
                            
                    if voter[j] >= 5:
                        client.publish(f'{topic_base}/{pose_dict[i]}', 1)
                        print('*'*10)
                        print(f'sent {pose_dict[i]}, 1')
                        
                        
                    elif voter[j] <= -5:
                        client.publish(f'{topic_base}/{pose_dict[i]}', -1)
                        print('*'*10)
                        print(f'sent {pose_dict[i]}, -1')
                        
                    
                    print('voter is:', voter)
                    print('iters is:', iters)
                    """
            #nose_to_lelbow = [lands[0], lands[13]]
            #nose_to_lwrist = [lands[0], lands[15]]
            
            #lflexor_xy_angle = max(0, (calculate_angle(ears, lshoulder_to_wrist, ['x', 'y'])))
            #rflexor_xy_angle = calculate_angle(ears, rshoulder_to_wrist, ['x', 'y'])

            #lflexor_xz_angle = max(0, (calculate_angle(ears, lshoulder_to_wrist, ['x', 'z'])))
            #lflexor_xz_angle = calc_magnitude_ratio(nose_to_lelbow, ears, ['x', 'z'])
            #lflexor_xz_angle = lshoulder_to_wrist[1].x*width - lshoulder_to_wrist[0].x*width
