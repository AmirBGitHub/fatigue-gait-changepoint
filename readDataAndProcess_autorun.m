% Release: $Name: Gait Segmentation 1.1 $
% $Revision: 1.1 $
% $Date: 2018-07-10 20:05:12 $

% Copyright (c) 2018, Amir Baghdadi.  All rights reserved.

clear all;close all;clc
SamplingRate = 51.2;
SubjHeight = [1.71 1.77 1.71 1.59 1.69 1.63 1.60 1.71 1.67 1.78 1.68 1.55 1.83 1.81 1.89];       

unzip(urlwrite('https://www.dropbox.com/s/fmxneqric3znfuv/data.zip?dl=1','data.zip'));
mkdir output 

for sbjct = 1:15
    
    %% data read, process and save segments
    Participant = sbjct
    m = sbjct; 
    clear imu_data stepDur ampRange lenPeak time M_i_k_filt 

    % Read raw data
    imu_read = importdata([pwd, '/data/subject_', num2str(m), '_raw.csv']);
    imu_data = imu_read.data;
    
    Data_in = imu_data(30720:end,1:6); 
    
    % read sample segments
    loadData = load(fullfile('./data/Test Seg', ['subject_', num2str(m), '_seg_S.mat']));
    M_i_k_S = loadData.M_i_k_S;
    loadData = load(fullfile('./data/Test Seg', ['subject_', num2str(m), '_seg_E.mat']));
    M_i_k_E = loadData.M_i_k_E;
    
    for c = 1:size(M_i_k_S)
        MIKSLng(c) = length(M_i_k_S{c,6});
        MIKSPk(c) = max(M_i_k_S{c,6});
    end
    for c = 1:size(M_i_k_E)
        MIKELng(c) = length(M_i_k_E{c,6});
        MIKEPk(c) = max(M_i_k_E{c,6});
    end
    
    StepLengthAvg = 50;
    PeakAccAvg = round(0.5*mean([MIKSPk,MIKEPk]));

    % Data processing and segmentation
    [M_i_k, Seg_points_S, Seg_points_E] = process_segment_3h(Data_in,SamplingRate,StepLengthAvg,PeakAccAvg);
    save(['./output/Subject', num2str(m),'.mat'],'M_i_k')

%     figure('Color',[1 1 1])
%     vidfile = VideoWriter(['SegmentationMovie',num2str(m),'_short.mp4'],'MPEG-4');
%     open(vidfile);
%     for k = 1:size(M_i_k,1)
%         plot(M_i_k{k,5},M_i_k{k,6},'r-','linewidth', 2)
%         xlabel('Time (sec)') % x-axis label
%         ylabel('Acceleration Magnitude') % y-axis label
%         xlim([0 1.5])
%         ylim([-25 25])
%         pause(0.05)
%         F(k) = getframe(gcf);
%         writeVideo(vidfile,F(k));
%     end
%     close(vidfile)

    %% filtering the outlier segments 
    for i = 1:size(M_i_k,1)
        time(i,1) = mean(M_i_k{i,17});
        stepDur(i,1) = range(M_i_k{i,3});
        lenPeak(i,1) = max(M_i_k{i,1});
        ampRange(i,1) = range(M_i_k{i,2});
    end
    featureX_filt = [ampRange/SubjHeight(m),lenPeak/SubjHeight(m),stepDur];
    figure;
    subplot(3,1,1)
    plot(featureX_filt(:,1),'r')
    xlabel('segments')
    ylabel('stide height')
    subplot(3,1,2)
    plot(featureX_filt(:,2),'g')
    xlabel('segments')
    ylabel('stide length')
    subplot(3,1,3)
    plot(featureX_filt(:,3),'b')
    xlabel('segments')
    ylabel('stide duration')

    saveas(gcf,['./output/Subject', num2str(m),'.jpg']);

end


function [M_i_k, Seg_points_S, Seg_points_E] = process_segment_3h(Data_in,SamplingRate,StepLengthAvg,PeakAccAvg)

    %% Read Data
    % This code imports the down sampled acceleration and gyro data and uses the butterworth low pass filter 
    % to smooth the transformed data from Kalman filter and segment the data into gait steps 
    acc_data = Data_in(:,1:3);
    gyro_data = Data_in(:,4:6);
    duration = (length(Data_in)-1)/SamplingRate;
    dt = 1/SamplingRate;
    t = (0:dt:duration)';
    %% Kalman filter 
    [State, Rot] = IMU_Kalman(Data_in, SamplingRate); 
    filtered_angle = State(:,7:9);      % filtered angle
    unbiased_gyro = State(:,10:12);     % unbiased gyro 
    % finds the quaternions of the rotation in order to transfering
    % body frame set of data to global frame
    acc_glob = Quaternion(acc_data,Rot);
    gyro_glob = Quaternion(gyro_data,Rot);
    acc_g = repmat([0 -9.81 0],length(acc_data),1);
    acc_lin = acc_glob + acc_g; 
    %% Segmentation and Kinematics Calculation
    % Butterworth filter parameters for acceleration 
    n=4;fc=4;Fs=100;
    Wn = (2/Fs)*fc;
    [b,a]=butter(n,Wn,'low');
    Acceleration_filt = filter(b,a,acc_lin);
    Acceleration_filt_magn = filter(b,a,sqrt(sum(Acceleration_filt.^2,2)));
    Acceleration_filt_magn = Acceleration_filt_magn + min(Acceleration_filt_magn);
    
    % Segmentation based on peaks in acc data
    Seg_points_S = [];
    Seg_points_E = [];
    
    n = 21;
    while n<length(t)-StepLengthAvg
        max_b = max(Acceleration_filt_magn(n+fix(StepLengthAvg/2):n+StepLengthAvg));
        max_a = max(Acceleration_filt_magn(n:n+fix(StepLengthAvg/2)));
        if (max_b/max_a>1.2 || max_a/max_b>1.2) && max(Acceleration_filt_magn(n:n+StepLengthAvg))>PeakAccAvg      
            [~,I1] = max(Acceleration_filt_magn(n:n+StepLengthAvg));
            if n+I1+20 <= length(Acceleration_filt_magn)  
                [~,I2] = min(Acceleration_filt_magn(n+I1:n+I1+20));
                [~,I3] = min(Acceleration_filt_magn(n-20:n+10));
                if abs(diff([n-20+I3, n+I1+I2])-StepLengthAvg) < 0.3*StepLengthAvg
                    %Seg_points_S = [Seg_points_S; n];
                    %Seg_points_E = [Seg_points_E; n+I1+I2];
                    Seg_points_S = [Seg_points_S; n-20+I3];
                    Seg_points_E = [Seg_points_E; n+I1+I2-2];
                    n = n + I1+I2;
                else
                    n = n + 1;
                end
            else
                n = n + 1;
            end
        else
            n = n + 1;
        end
    end
    
    % Making segments windows
    k = 1;
    for j = 2:length(Seg_points_E)
        S_s(k) = Seg_points_S(j);
        S_e(k) = Seg_points_E(j);
        k = k + 1;
    end     

    axis = [1 2 3];

    for direc = axis

        % Velocity calculation
        Vel{1,direc} = zeros(length(t),1);
        Res_acc = zeros(length(S_e),1);
        for i = 1:length(S_s)-1
            Res_acc(i,1) = trapz(t(S_s(i):S_e(i)),Acceleration_filt(S_s(i):S_e(i),direc))/range(t(S_s(i):S_e(i))); 
            for j = S_s(i):S_e(i)
                Vel{1,direc}(j,:) = trapz(t(S_s(i):j+1),Acceleration_filt(S_s(i):j+1,direc)) - Res_acc(i,1)*range(t(S_s(i):j+1));
            end
        end

        % Position calculation
        for i = 1:length(t)-1
            Pos{1,direc}(i+1,:) = trapz(t(1:i+1),(Vel{1,direc}(1:i+1)));
        end

        % Jerk calculation
        for i = 1:length(t)-1
            Jrk{1,direc}(i,:) = diff(Acceleration_filt(i:i+1,direc))/dt;
        end

    end

    % Metrics Segmentation
    Velocity = mag(cell2mat(Vel),1);
    Jerk = mag(cell2mat(Jrk),1);
    
    m = [Pos{1,1}(2:end), Pos{1,2}(2:end), Pos{1,3}(2:end), t(2:end), Velocity(2:end), t(2:end), Acceleration_filt_magn(2:end), t(2:end), Jerk, filtered_angle(2:end,1), filtered_angle(2:end,2), filtered_angle(2:end,1), unbiased_gyro(2:end,1), filtered_angle(2:end,2), unbiased_gyro(2:end,2),filtered_angle(2:end,3), unbiased_gyro(2:end,3), t(2:end)];
    
    SegStep = Seg_points_E-Seg_points_S;
    
    for i = 1:size(m,2)
        k = 1;
        for j = 1:length(SegStep)-1
            %if SegStep(j)<90 && SegStep(j)>4
                % Making the data start from zero in each segment window
                if i == size(m,2) 
                    M_i_k{k,i} = m(S_s(j):S_e(j),i);
                else
                    M_i_k{k,i} = m(S_s(j):S_e(j),i) - m(S_s(j),i);
                end
                k = k + 1;
            %end
        end
    end
    
    
    for i = 1:length(M_i_k(:,1))
        x = M_i_k{i,1};
        y = M_i_k{i,2};
        z = M_i_k{i,3};
        
        sf = fit([x, y],z,'poly11');
%         plot(sf,[x,y],z)
%         axis equal
        
        x_SP = ([x(end),y(end),z(end)]-[x(1),y(1),z(1)])./norm([x(end),y(end),z(end)]-[x(1),y(1),z(1)]);
        z_SP = [sf.p10,sf.p01,-1]./norm([sf.p10,sf.p01,-1]);  
        y_SP = cross(x_SP,z_SP);
%         hold on
%         plot3(linspace(0,x_SP(1),10),linspace(0,x_SP(2),10),linspace(0,x_SP(3),10),'r*','linewidth',5)
%         hold on
%         plot3(linspace(0,y_SP(1),10),linspace(0,y_SP(2),10),linspace(0,y_SP(3),10),'r*','linewidth',5)
%         hold on
%         plot3(linspace(0,z_SP(1),10),linspace(0,z_SP(2),10),linspace(0,z_SP(3),10),'r*','linewidth',5)

        M = [x_SP;y_SP;z_SP];
        xyz_SP = M*[x,y,z]';
% 
%         figure;
%         plot(xyz_SP(1,:),xyz_SP(2,:))
        
        M_i_k{i,1} = 2*xyz_SP(1,:);
        M_i_k{i,2} = xyz_SP(2,:);
    end
    
    M_i_k(:,3)=[];    
end


function X_glob = Quaternion(X_body,R)

Q = dcm2q(R);
X_glob = qvqc(Q,X_body);

%% Functions

    function q=dcm2q(R)
        % DCM2Q(R) converts direction cosine matrices into quaternions.
        %
        %     The resultant quaternion(s) will perform the equivalent vector
        %     transformation as the input DCM(s), i.e.:
        %
        %       qconj(Q)*V*Q = R*V
        %
        %     where R is the DCM, V is a vector, and Q is the quaternion.  Note that
        %     for purposes of quaternion-vector multiplication, a vector is treated
        %     as a quaterion with a scalar element of zero.
        %
        %     If the input is a 3x3xN array, the output will be a vector of
        %     quaternions where input direction cosine matrix R(:,:,k) corresponds
        %     to the output quaternion Q(k,:).
        %
        %     Note that this function is meaningless for non-orthonormal matrices!
        %
        % See also Q2DCM.

        % Release: $Name: quaternions-1_3 $
        % $Revision: 1.11 $
        % $Date: 2009-07-25 04:28:18 $

        % Copyright (c) 2000-2009, Jay A. St. Pierre.  All rights reserved.

        % Thanks to Tatsuki Kashitani for pointing out the numerical instability in
        % the original implementation.  His suggested fix also included a check for
        % the "sr" values below being zero.  But I think I've convinced myself that
        % this isn't necessary if the input matrices are orthonormal (or at least
        % very close to orthonormal).

        if nargin~=1
          error('One input argument required');
        else
          size_R=size(R);
          if ( size_R(1)~=3 || size_R(2)~=3 || length(size_R)>3 )
            error('Invalid input: must be a 3x3xN array')
          end
        end

        q = zeros( 4, size( R, 3 ) );

        for id_dcm = 1 : size( R, 3 )
          dcm = R( :, :, id_dcm );
          if trace( dcm ) > 0
            % Positve Trace Algorithm
            sr  = sqrt( 1 + trace( dcm ));
            sr2 = 2*sr;
            q(1,id_dcm) = ( dcm(2,3) - dcm(3,2) ) / sr2;
            q(2,id_dcm) = ( dcm(3,1) - dcm(1,3) ) / sr2;
            q(3,id_dcm) = ( dcm(1,2) - dcm(2,1) ) / sr2;
            q(4,id_dcm) = 0.5 * sr;
          else
            % Negative Trace Algorithm
            if ( dcm(1,1) > dcm(2,2) ) && ( dcm(1,1) > dcm(3,3) )
              % Maximum Value at DCM(1,1)
              sr  = sqrt( 1 + (dcm(1,1) - ( dcm(2,2) + dcm(3,3) )) );
              sr2 = 2*sr;
              q(1,id_dcm) = 0.5 * sr;
              q(2,id_dcm) = ( dcm(2,1) + dcm(1,2) ) / sr2;
              q(3,id_dcm) = ( dcm(3,1) + dcm(1,3) ) / sr2;
              q(4,id_dcm) = ( dcm(2,3) - dcm(3,2) ) / sr2;
            elseif dcm(2,2) > dcm(3,3)
              % Maximum Value at DCM(2,2)
              sr  = sqrt( 1 + (dcm(2,2) - ( dcm(3,3) + dcm(1,1) )) );
              sr2 = 2*sr;
              q(1,id_dcm) = ( dcm(2,1) + dcm(1,2) ) / sr2;
              q(2,id_dcm) = 0.5 * sr;
              q(3,id_dcm) = ( dcm(2,3) + dcm(3,2) ) / sr2;
              q(4,id_dcm) = ( dcm(3,1) - dcm(1,3) ) / sr2;
            else
              % Maximum Value at DCM(3,3)
              sr  = sqrt( 1 + (dcm(3,3) - ( dcm(1,1) + dcm(2,2) )) );
              sr2 = 2*sr;
              q(1,id_dcm) = ( dcm(3,1) + dcm(1,3) ) / sr2;
              q(2,id_dcm) = ( dcm(2,3) + dcm(3,2) ) / sr2;
              q(3,id_dcm) = 0.5 * sr;
              q(4,id_dcm) = ( dcm(1,2) - dcm(2,1) ) / sr2;
            end
          end
        end

        % Make quaternion vector a column of quaternions
        q=q.';

        q=real(q);
    end


    function qtype=isq(q)
        % ISQ(Q) checks to see if Q is a quaternion or set of quaternions.
        %     ISQ returns a value accordingly:
        %
        %        0 if Q is not a quaternion or vector of quaternions:
        %          has more than 2 dimensions or neither dimension is of length 4
        %       
        %        1 if the component quaternions of Q are column vectors:
        %          Q is 4xN, where N~=4, or
        %          Q is 4x4 and only the columns are normalized 
        %
        %        2 if the component quaternions of Q are row vectors:
        %          Q is Nx4, where N~=4, or
        %          Q is 4x4 and only the rows are normalized 
        %
        %        3 if the shape of the component quaternions is indeterminant:
        %          Q is 4x4, and either both the columns and rows are normalized
        %          or neither the columns nor rows are normalized.
        %
        %     In other words, if Q is 4x4, ISQ attempts to discern the shape of
        %     component quaternions by determining whether the rows or the columns
        %     are normalized (i.e., it assumes that normalized quaternions are
        %     the more typical use of quaternions).
        %
        %     The test for normalization uses 2*EPS as a tolerance.
        %
        % See also ISNORMQ, EPS.

        % Release: $Name: quaternions-1_3 $
        % $Revision: 1.7 $
        % $Date: 2009-07-26 20:05:12 $

        % Copyright (c) 2001-2009, Jay A. St. Pierre.  All rights reserved.

        if nargin~=1

          error('isq() requires one input argument');

        else

          tol=2*eps;

          size_q=size(q);

          if ( length(size_q)~=2 || max(size_q==4)~=1 )
            qtype=0; % Not a quaternion or quaternion vector

          elseif ( size_q(1)==4 && ...
                   ( size_q(2)~=4 || ( ~sum((sum(q.^2,1)-ones(1,4))>tol) &&   ...
                                        sum((sum(q.^2,2)-ones(4,1))>tol)    ) ...
                     ) ...
                   )
            qtype=1; % Component q's are column vectors

          elseif ( size_q(2)==4 && ...
                   ( size_q(1)~=4 || ( ~sum((sum(q.^2,2)-ones(4,1))>tol) &&   ...
                                        sum((sum(q.^2,1)-ones(1,4))>tol)    ) ...
                     ) ...
                   )
            qtype=2; % Component q's are row vectors

          else
            qtype=3; % Component q's are either columns or rows (indeterminate)

          end

        end
    end


    function qout=qconj(qin)
        % QCONJ(Q) calculates the conjugate of the quaternion Q.
        %     Works on "vectors" of quaterions as well.  Will return the same shape
        %     vector as input.  If input is a vector of four quaternions, QCONJ will
        %     determine whether the quaternions are row or column vectors according
        %     to ISQ.
        %
        % See also ISQ.

        % Release: $Name: quaternions-1_3 $
        % $Revision: 1.16 $
        % $Date: 2009-07-26 20:05:12 $

        % Copyright (c) 2001-2009, Jay A. St. Pierre.  All rights reserved.


        if nargin~=1
          error('qconj() requires one input argument');
        else
          qtype = isq(qin);
          if ( qtype==0 )
            error(...
              'Invalid input: must be a quaternion or a vector of quaternions')
          elseif ( qtype==3 )
            warning(...
              'qconj:indeterminateShape', ...
              'Component quaternion shape indeterminate, assuming row vectors')
          end
        end

        % Make sure component quaternions are row vectors
        if( qtype == 1 )
          qin=qin.';
        end

        qout(:,1)=-qin(:,1);
        qout(:,2)=-qin(:,2);
        qout(:,3)=-qin(:,3);
        qout(:,4)= qin(:,4);

        % Make sure output is same shape as input
        if( qtype == 1 )
          qout=qout.';
        end
    end

    function v_out=qcvq(q,v)
        % QcVQ(Q,V) performs the operation qconj(Q)*V*Q
        %     where the vector is treated as a quaternion with a scalar element of
        %     zero.
        %
        %     Q and V can be vectors of quaternions and vectors, but they must
        %     either be the same length or one of them must have a length of one.
        %     The output will have the same shape as V.  Q will be passed through
        %     QNORM to ensure it is normalized.
        %
        % See also QVQc, QNORM, QMULT.

        % Note that QNORM is invoked by QMULT, therefore QcQV does not invoke
        % it directly.

        % Release: $Name: quaternions-1_3 $
        % $Revision: 1.2 $
        % $Date: 2009-07-26 20:05:12 $

        % Copyright (c) 2000-2009, Jay A. St. Pierre.  All rights reserved.


        if nargin~=2
          error('Two input arguments required.');
        else

          qtype=isq(q);
          if ( qtype == 0 )
            error('Input Q must be a quaternion or a vector of quaternions')
          end

          size_v=size(v);
          if ( length(size_v)~=2 || max(size_v==3)~=1 )
            error(['Invalid input: second input must be a 3-element vector', 10, ...
                   'or a vector of 3-element vectors'])
          end

        end

        % Make sure q is a column of quaternions
        if ( qtype==1 )
          q=q.';
        end

        % Make sure v is a column of vectors
        row_of_vectors = (size_v(2) ~= 3);
        if ( row_of_vectors )
          v=v.';
          size_v=size(v);
        end

        size_q=size(q);

        if (  size_q(1)~=size_v(1) && size_q(1)~=1 && size_v(1)~=1 )
          error(['Inputs do not have the same number of elements:', 10, ...
                 '   number of quaternions in q = ', num2str(size_q(1)), 10,...
                 '   number of vectors in v     = ', num2str(size_v(1)), 10,...
                 'Inputs must have the same number of elements, or', 10, ...
                 'one of the inputs must have a single element.']) 
        elseif ( size_q(1)==1 && size_v(1)==3 )
          if ( qtype==1 )
            warning(...
              'qcvq:assumingVcols', ...
              'Q is 4x1 and V is 3x3: assuming vectors are column vectors')
            row_of_vectors = 1;
            v=v.';
          else
            warning(...
              'qcvq:assumingVrows', ...
              'Q is 1x4 and V is 3x3: assuming vectors are row vectors')
          end
        elseif ( qtype==3 && size_v(1)==1 )
          if ( row_of_vectors )
            warning(...
              'qcvq:assumingQcols', ...
              'Q is 4x4 and V is 3x1: assuming quaternions are column vectors')
            q=q.';
          else
            warning(...
              'qcvq:assumingQrows', ...
              'Q is 4x4 and V is 1x3: assuming quaternions are row vectors')
          end  
        end

        % Build up full vectors if one input is a singleton
        if ( size_q(1) ~= size_v(1) )
          ones_length = ones(max(size_q(1),size_v(1)),1);
          if ( size_q(1) == 1 )
            q = [q(1)*ones_length ...
                 q(2)*ones_length ...
                 q(3)*ones_length ...
                 q(4)*ones_length ];
          else % size_v(1) == 1
            v = [v(1)*ones_length ...
                 v(2)*ones_length ...
                 v(3)*ones_length ];    
          end
        end

        % Add an element to V
        v(:,4)=zeros(size_v(1),1);

        % Turn off warnings before calling qconj (it has simillar warnings as
        % qvxform, so all warnings would just be duplicated).  Save current state of
        % warnings, though.
        warning_state = warning; warning('off', 'qconj:indeterminateShape');
        local_warning = lastwarn;

        % Perform transform
        vt=qmult(qconj(q),qmult(v,q));

        % Restore warning state to original state
        warning(warning_state);
        lastwarn(local_warning);

        % Eliminate last element of vt for output
        v_out = vt(:,1:3);

        % Make sure output vectors are the same shape as input vectors
        if ( row_of_vectors )
          v_out = v_out.';
        end
    end
    
    
    function q_out=qmult(q1,q2)
        % QMULT(Q1,Q2) calculates the product of two quaternions Q1 and Q2.
        %    Inputs can be vectors of quaternions, but they must either have the
        %    same number of component quaternions, or one input must be a single
        %    quaternion.  QMULT will determine whether the component quaternions of
        %    the inputs are row or column vectors according to ISQ.
        %  
        %    The output will have the same shape as Q1.  If the component
        %    quaternions of either Q1 or Q2 (but not both) are of indeterminate
        %    shape (see ISQ), then the shapes will be assumed to be the same for
        %    both inputs.  If both Q1 and Q2 are of indeterminate shape, then both
        %    are assumed to be composed of row vector quaternions.
        %
        % See also ISQ.

        % Release: $Name: quaternions-1_3 $
        % $Revision: 1.14 $
        % $Date: 2009-07-26 20:05:12 $

        % Copyright (c) 2001-2009, Jay A. St. Pierre.  All rights reserved.


        if nargin~=2
          error('qmult() requires two input arguments');
        else
          q1type = isq(q1);
          if ( q1type == 0 )
            error(['Invalid input: q1 must be a quaternion or a vector of' ...
                  ' quaternions'])
          end
          q2type = isq(q2);
          if ( q2type == 0 )
            error(['Invalid input: q2 must be a quaternion or a vector of' ...
                  ' quaternions'])
          end
        end

        % Make sure q1 is a column of quaternions (components are rows)
        if ( q1type==1 || (q1type==3 && q2type==1) )
          q1=q1.';
        end

        % Make sure q2 is a column of quaternions (components are rows)
        if ( q2type==1 || (q2type==3 && q1type==1) )
          q2=q2.';
        end

        num_q1=size(q1,1);
        num_q2=size(q2,1);

        if (  num_q1~=num_q2 && num_q1~=1 && num_q2~=1 )
          error(['Inputs do not have the same number of elements:', 10, ...
                 '   number of quaternions in q1 = ', num2str(num_q1), 10,...
                 '   number of quaternions in q2 = ', num2str(num_q2), 10,...
                 'Inputs must have the same number of elements, or', 10, ...
                 'one of the inputs must be a single quaternion (not a', 10, ...
                 'vector of quaternions).']) 
        end

        % Build up full quaternion vector if one input is a single quaternion
        if ( num_q1 ~= num_q2 )
          ones_length = ones(max(num_q1,num_q2),1);
          if ( num_q1 == 1 )
            q1 = [q1(1)*ones_length ...
                  q1(2)*ones_length ...
                  q1(3)*ones_length ...
                  q1(4)*ones_length ];
          else % num_q2 == 1
            q2 = [q2(1)*ones_length ...
                  q2(2)*ones_length ...
                  q2(3)*ones_length ...
                  q2(4)*ones_length ];    
          end
        end

        % Products

        % If q1 and q2 are not vectors of quaternions, then:
        %
        %   q1*q2 = q1*[ q2(4) -q2(3)  q2(2) -q2(1)
        %                q2(3)  q2(4) -q2(1) -q2(2)
        %               -q2(2)  q2(1)  q2(4) -q2(3)
        %                q2(1)  q2(2)  q2(3)  q2(4) ]
        %
        % But to deal with vectorized quaternions, we have to use the ugly
        % commands below.

        prod1 = ...
            [ q1(:,1).*q2(:,4) -q1(:,1).*q2(:,3)  q1(:,1).*q2(:,2) -q1(:,1).*q2(:,1)];
        prod2 = ...
            [ q1(:,2).*q2(:,3)  q1(:,2).*q2(:,4) -q1(:,2).*q2(:,1) -q1(:,2).*q2(:,2)];
        prod3 = ...
            [-q1(:,3).*q2(:,2)  q1(:,3).*q2(:,1)  q1(:,3).*q2(:,4) -q1(:,3).*q2(:,3)];
        prod4 = ...
            [ q1(:,4).*q2(:,1)  q1(:,4).*q2(:,2)  q1(:,4).*q2(:,3)  q1(:,4).*q2(:,4)];

        q_out = prod1 + prod2 + prod3 + prod4;

        % Make sure output is same format as q1
        if ( q1type==1 || (q1type==3 && q2type==1) )
          q_out=q_out.';
        end

        % NOTE that the following algorithm proved to be slower than the one used
        % above:
        %
        % q_out = zeros(size(q1));
        % 
        % q_out(:,1:3) = ...
        %     [q1(:,4) q1(:,4) q1(:,4)].*q2(:,1:3) + ...
        %     [q2(:,4) q2(:,4) q2(:,4)].*q1(:,1:3) + ...
        %     cross(q1(:,1:3), q2(:,1:3));
        % 
        % q_out(:,4) = q1(:,4).*q2(:,4) - dot(q1(:,1:3), q2(:,1:3), 2);
    end

    
    function v_out=qvqc(q,v)
        % QVQc(Q,V) performs the operation Q*V*qconj(Q)
        %     where the vector is treated as a quaternion with a scalar element of
        %     zero. 
        %
        %     Q and V can be vectors of quaternions and vectors, but they must
        %     either be the same length or one of them must have a length of one.
        %     The output will have the same shape as V.  Q will be passed through
        %     QNORM to ensure it is normalized.
        %
        % See also QcQV, QNORM

        % Release: $Name: quaternions-1_3 $
        % $Revision: 1.1 $
        % $Date: 2009-07-24 19:14:44 $

        % Copyright (c) 2000-2009, Jay A. St. Pierre.  All rights reserved.


        if nargin~=2
          error('Two input arguments required');
        else
          q     = qconj(q);
          v_out = qcvq(q, v);
        end
    end
    
end


function [State, R] = IMU_Kalman(imu_data, SamplingRate)

duration = (length(imu_data)-1)/SamplingRate;
dt = 1/SamplingRate;
t = (0:dt:duration)';

acc_data = imu_data(:,1:3);
gyro_data = imu_data(:,4:6);
gyro = gyro_data;

v_gyro = sqrt(0.3)*randn(length(t),3); % measurement noise for gyro,  variance = 0.3
v_acc = sqrt(0.3)*randn(length(t),3); % measurement noise for accellerometer, variance = 0.3

% Compute the angles computed by using only accelerometers of gyroscope
x = [atan(acc_data(:,1)./sqrt(sum([acc_data(:,2).^2 acc_data(:,3).^2],2)))*(180/pi), ...
    atan(acc_data(:,2)./sqrt(sum([acc_data(:,1).^2 acc_data(:,3).^2],2)))*(180/pi), ...
    atan(acc_data(:,3)./sqrt(sum([acc_data(:,1).^2 acc_data(:,2).^2],2)))*(180/pi)];
x_acc = x + v_acc; % Angle computed from accelerometer measurement
% x_acc = x;
x_gyro = cumsum(gyro*dt,1); % Angle computed by integration of gyro measurement

P = [1 0; 0 1];
R_angle = 0.3;

Q_angle = 0.05;
Q_gyro = 0.5;
Q = [Q_angle 0; 0 Q_gyro];

A = [0 -1; 0 0];
q_bias = [0 0 0]; % Initialize gyro bias
angle = [0 90 0]; % Initialize gyro angle
q_m = 0;
X = [0 90 0; 0 0 0];

    for i=1:length(t)

         % Gyro update 

         q_m = gyro(i,:);

         q = q_m - q_bias; % gyro bias removal

         Pdot = A*P + P*A' + Q;

         rate = q;

         angle = angle + q*dt;

         P = P + Pdot*dt;

         % Kalman (Accelerometer) update 

         C = [1 0];
         angle_err = x_acc(i,:)-angle;
         E = C*P*C' + R_angle;

         K = P*C'*inv(E);

         P = P - K*C*P;
         X = X + K * angle_err;
         x1(i,:) = X(1,:);
         R(:,:,i) = ang2orth(x1);
         x2(i,:) = X(2,:);
         angle = x1(i,:);
         q_bias = x2(i,:);

         x3(i,:) = q;  % unbiased gyro rate
    end

State = [x_acc x_gyro x1 x3];

x_acc = State(:,1:3);   % angle calculated from accelerometer
x_gyro = State(:,4:6);  % angle calculated from gyro 
filtered_angle = State(:,7:9);      % filtered angle
unbiased_gyro = State(:,10:12);     % unbiased gyro
rotation_mat = R;


% %Plot the state before using Kalman filter
% figure;
% subplot(3,4,1);
% plot(t,x_acc(:,1));
% xlabel('time(s)');
% ylabel('Theta x(t)');
% legend('acc angle');
% 
% subplot(3,4,5);
% plot(t,x_acc(:,2));
% xlabel('time(s)');
% ylabel('Theta y(t)');
% legend('acc angle');
% 
% subplot(3,4,9);
% plot(t,x_acc(:,3));
% xlabel('time(s)');
% ylabel('Theta z(t)');
% legend('acc angle');
% 
% subplot(3,4,2);
% plot(t,x_gyro(:,1));
% xlabel('time(s)');
% ylabel('Theta x(t)');
% legend('gyro angle');
% 
% subplot(3,4,6);
% plot(t,x_gyro(:,2));
% xlabel('time(s)');
% ylabel('Theta y(t)');
% legend('gyro angle');
% 
% subplot(3,4,10);
% plot(t,x_gyro(:,3));
% xlabel('time(s)');
% ylabel('Theta z(t)');
% legend('gyro angle');
% 
% % Plot the result using kalman filter
% subplot(3,4,3);
% plot(t,unbiased_gyro(:,1));
% xlabel('time(s)');
% ylabel('ThetaDot x(t)');
% legend('gyro rate unbiased');
% 
% subplot(3,4,7);
% plot(t,unbiased_gyro(:,2));
% xlabel('time(s)');
% ylabel('ThetaDot y(t)');
% legend('gyro rate unbiased');
% 
% subplot(3,4,11);
% plot(t,unbiased_gyro(:,3));
% xlabel('time(s)');
% ylabel('ThetaDot z(t)');
% legend('gyro rate unbiased');
% 
% subplot(3,4,4);
% plot(t,filtered_angle(:,1));
% xlabel('time(s)');
% ylabel('Theta x(t)');
% legend('kalman angle');
% 
% subplot(3,4,8);
% plot(t,filtered_angle(:,2));
% xlabel('time(s)');
% ylabel('Theta y(t)');
% legend('kalman angle');
% 
% subplot(3,4,12);
% plot(t,filtered_angle(:,3));
% xlabel('time(s)');
% ylabel('Theta z(t)');
% legend('kalman angle');

    function orthm = ang2orth(ang) 
        sa = sind(ang(2)); ca = cosd(ang(2)); 
        sb = sind(ang(1)); cb = cosd(ang(1)); 
        sc = sind(ang(3)); cc = cosd(ang(3)); 

        ra = [  ca,  sa,  0; ... 
               -sa,  ca,  0; ... 
                 0,   0,  1]; 
        rb = [  cb,  0,  sb; ... 
                 0,  1,  0; ... 
               -sb,  0,  cb]; 
        rc = [  1,   0,   0; ... 
                0,   cc, sc;... 
                0,  -sc, cc]; 
        orthm = rc*rb*ra;
    end

end


function N = mag(T,n)
% MAGNATUDE OF A VECTOR (Nx3)
%  M = mag(U)
N = sum(abs(T).^2,2).^(1/2);
d = find(N==0); 
N(d) = eps*ones(size(d));
N = N(:,ones(n,1));  
end
