classdef Net < handle
  % Wrapper class of caffe::Net in matlab
  
  properties (Access = private)
    hNet_self
    attributes
    % attribute fields
    %     hLayer_layers
    %     hBlob_blobs
    %     input_blob_indices
    %     output_blob_indices
    %     layer_names
    %     blob_names
  end
  properties (SetAccess = private)
    layer_vec
    blob_vec
    inputs
    outputs
    name2layer_index
    name2blob_index
    layer_names
    blob_names
    mean
  end
  
  methods
    function self = Net(varargin)
      % decide whether to construct a net from model_file or handle
      if ~(nargin == 1 && isstruct(varargin{1}))
        % construct a net from model_file
        self = caffe.get_net(varargin{:});
        return
      end
      % construct a net from handle
      hNet_net = varargin{1};
      CHECK(is_valid_handle(hNet_net), 'invalid Net handle');
      
      % setup self handle and attributes
      self.hNet_self = hNet_net;
      self.attributes = caffe_('net_get_attr', self.hNet_self);
      
      % setup layer_vec
      self.layer_vec = caffe.Layer.empty();
      for n = 1:length(self.attributes.hLayer_layers)
        self.layer_vec(n) = caffe.Layer(self.attributes.hLayer_layers(n));
      end
      
      % setup blob_vec
      self.blob_vec = caffe.Blob.empty();
      for n = 1:length(self.attributes.hBlob_blobs);
        self.blob_vec(n) = caffe.Blob(self.attributes.hBlob_blobs(n));
      end
      
      % setup input and output blob and their names
      % note: add 1 to indices as matlab is 1-indexed while C++ is 0-indexed
      self.inputs = ...
        self.attributes.blob_names(self.attributes.input_blob_indices + 1);
      self.outputs = ...
        self.attributes.blob_names(self.attributes.output_blob_indices + 1);
      
      % create map objects to map from name to layers and blobs
      self.name2layer_index = containers.Map(self.attributes.layer_names, ...
        1:length(self.attributes.layer_names));
      self.name2blob_index = containers.Map(self.attributes.blob_names, ...
        1:length(self.attributes.blob_names));
      
      % expose layer_names and blob_names for public read access
      self.layer_names = self.attributes.layer_names;
      self.blob_names = self.attributes.blob_names;
        
      if isfield(self.attributes,'mean')
          self.mean = self.attributes.mean;
      else
          self.mean = [];
      end
    end

    function [features,blob_ids] = get_features(self, images, layer)
	%CAFFE_FEATURES Calculates the intermediate activations of a cnn for all 
	% images in the filelist. 
	% @param images Path to text file containing a list of paths to images or
	% cell array containing a list of image arrays
	% @param layer    One of the layer of the CNN, check the prototxt file for
	% their names
	    if nargin<3
            layer='relu7';
        end
        
        use_vl = exist('vl_imreadjpeg')==3;
            
	    layer_id = self.name2layer_index(layer);
	    batch_size=self.blob_vec(1).num();
	    
	    filelistmode=ischar(images)||( iscell(images) && ischar(images{1}));
	    if ischar(images)
            % load the file list
            fid=fopen(images);
            fl=textscan(fid,'%s');
            fl=fl{1};
            fclose(fid);
        else
            fl=images;
	    end
	    % create tmp for batch
	    batch_data = zeros(self.blob_vec(1).shape(),'single');
	    % Calculate the starting indices of every batch
	    slices=1:batch_size:size(fl,1);
	    slices(end+1)=size(fl,1)+1;
	    print_count = 0;
        if filelistmode && use_vl
            % Prefetch the first batch already
            vl_imreadjpeg(fl(slices(1):slices(2)-1), 'prefetch');
        end
	    % for every slice
	    for i=1:numel(slices)-1
            if (i>1 && mod(i,10)==0)
                fprintf(repmat('\b',1,print_count));
                print_count = fprintf('Running batch number %i of %i\n',i, numel(slices)-1);
            end
            % load the image of the slice
            ims = fl(slices(i):slices(i+1)-1);
            if (filelistmode)
                if use_vl
                    % Get the images from last prefetch
                    ims = vl_imreadjpeg(ims);
                    % Start next prefetch
                    if i<numel(slices)-1
                        vl_imreadjpeg(fl(slices(i+1):slices(i+2)-1), 'prefetch');
                    end
                else
                    ims = cellfun(@imread,ims,'UniformOutput',false);
                end
            end
            ims = cellfun(@self.caffe_prepare_image,ims,'UniformOutput',false);
            batch_data(:,:,:,1:numel(ims)) = cat(4,ims{:});

            self.blobs(self.inputs{1}).set_data(batch_data);
            blob_ids = self.forward_prefilled(1,layer_id);
            tmp_feat = self.blob_vec(blob_ids(1)).get_data();
            tmp_feat = reshape(tmp_feat,[],batch_size)';
            if (~exist('features','var'))
                features = zeros(size(fl,1),size(tmp_feat,2),'single');
            end
            features(slices(i):(slices(i+1)-1),:)=tmp_feat(1:(slices(i+1)-slices(i)),:);
        end
        fprintf(repmat('\b',1,print_count));
    end

    function [ gradients ] = get_gradients(self, im, layer, channels )
        %CAFFE_GRADIENTS Calculates the gradients of a intermediate layer output
        % with respect the CNN input (that is the input image). This function adds
        % the gradients of all elements of the same channel implictly.
        % @param im       A regular image, that was read using the imread(..)
        % function. 
        % @param layer    One of the layer of the CNN, check the prototxt file for
        % their names.
        % @channels       The ids of the channels, that you want to calculate the
        % gradients from. The ids start at 1! You should make, that the ids do not
        % exceed the number of channels in your selected layer. In that case, the
        % behavior is undefined. 

        if nargin < 3
            layer = 'pool5';
        end
        layer_id = self.name2layer_index(layer);
        im = self.caffe_prepare_image(im);
        batch_size = self.blobs(self.inputs{1}).num();
        im = repmat(im,1,1,1,batch_size);
        self.blobs(self.inputs{1}).set_data(im);
        blob_ids = self.forward_prefilled();
        blob_id = 15;%blob_ids(1);
        
        num_channels = self.blob_vec(blob_id).num();
        if nargin<4
            channels = (1:num_channels)';
        end
        if isempty(self.name2layer_index(layer))
          error('A layer with this name does not exist.')
        end

        % Calc the gradients
        input_dims = self.blobs(self.inputs{1}).shape();
        gradients = zeros([input_dims(1:3) numel(channels)],'single');
	    % Calculate the starting indices of every batch
	    slices=1:batch_size:numel(channels);
	    slices(end+1)=numel(channels)+1;
	    print_count = 0;
	    for i=1:numel(slices)-1
            if (i>1 && mod(i,10)==0)
                fprintf(repmat('\b',1,print_count));
                print_count = fprintf('Running batch number %i of %i\n',i, numel(slices)-1);
            end
            % load the image of the next slice
            blob = self.blob_vec(blob_id);
            blob_diff = zeros(blob.width(),blob.height(),blob.channels(),blob.num(),'single');
            blob_idx = 1;
            for j=slices(i):slices(i+1)-1;
                blob_diff(:,:,channels(j),blob_idx)=1;
                blob_idx = blob_idx + 1;
            end
            self.blob_vec(blob_id).set_diff(reshape(blob_diff,blob.shape()));
            self.backward_prefilled();
            gradients(:,:,:,slices(i):(slices(i+1)-1)) = self.blobs(self.inputs{1}).get_diff();
        end
        
    end

    function [ im ] = caffe_prepare_image(self, im, stretch )
        if nargin<3
            stretch = false;
        end
        interpolation_type = 'bilinear';
        width=self.blob_vec(1).width();
        height=self.blob_vec(1).height();
        % make sure it's single type
        im = single(im);
        % resize to mean image
        if stretch
            im = imresize(im,[height width]);
        else
            if size(im,1)/size(im,2)<height/width
                % Scale the width to target width
                im = imresize(im,[NaN width],interpolation_type);
            else
                % Scale the height to target height
                im = imresize(im,[height NaN],interpolation_type);
            end
        end
        
        % catch gray scale images
        if (size(im,3)==1)
            im=repmat(im,1,1,3);
        else
            im = im(:,:,[3 2 1]);
        end
        
        if ~isempty(self.mean)
            % subtract with center crop of mean
            offset_row=int32(size(self.mean,1)-size(im,1))/2+1;
            offset_col=int32(size(self.mean,2)-size(im,2))/2+1;
            im = im - self.mean(offset_row:offset_row+size(im,1)-1,offset_col:offset_col+size(im,2)-1,:);
        end
        
        %  transpose 
        im = permute(im, [2 1 3]);
        
        if ~stretch
            % Embed image into target size
            tmp = zeros(width,height,self.blob_vec(1).channels(),'single');
            offset_row=int32(height-size(im,1))/2+1;
            offset_col=int32(width-size(im,2))/2+1;
            % Embed image into target
            tmp(offset_row:offset_row+size(im,1)-1,offset_col:offset_col+size(im,2)-1,:) = im;
            im = tmp;
        end
    end


    function layer = layers(self, layer_name)
      CHECK(ischar(layer_name), 'layer_name must be a string');
      layer = self.layer_vec(self.name2layer_index(layer_name));
    end
    function blob = blobs(self, blob_name)
      CHECK(ischar(blob_name), 'blob_name must be a string');
      blob = self.blob_vec(self.name2blob_index(blob_name));
    end
    function blob = params(self, layer_name, blob_index)
      CHECK(ischar(layer_name), 'layer_name must be a string');
      CHECK(isscalar(blob_index), 'blob_index must be a scalar');
      blob = self.layer_vec(self.name2layer_index(layer_name)).params(blob_index);
    end
    function blob_ids = forward_prefilled(self, from_layer, to_layer)
      if nargin>2 && from_layer>0 && to_layer>0
        blob_ids = caffe_('net_forward', self.hNet_self, from_layer, to_layer)+1;
      else
        caffe_('net_forward', self.hNet_self);
        blob_ids = [];
      end
    end
    function backward_prefilled(self, from_layer, to_layer)
      if nargin>2 
        caffe_('net_backward', self.hNet_self, from_layer, to_layer);
      else
        caffe_('net_backward', self.hNet_self);
      end
    end
    function res = forward(self, input_data)
      CHECK(iscell(input_data), 'input_data must be a cell array');
      CHECK(length(input_data) == length(self.inputs), ...
        'input data cell length must match input blob number');
      % copy data to input blobs
      for n = 1:length(self.inputs)
        self.blobs(self.inputs{n}).set_data(input_data{n});
      end
      self.forward_prefilled();
      % retrieve data from output blobs
      res = cell(length(self.outputs), 1);
      for n = 1:length(self.outputs)
        res{n} = self.blobs(self.outputs{n}).get_data();
      end
    end
    function res = backward(self, output_diff)
      CHECK(iscell(output_diff), 'output_diff must be a cell array');
      CHECK(length(output_diff) == length(self.outputs), ...
        'output diff cell length must match output blob number');
      % copy diff to output blobs
      for n = 1:length(self.outputs)
        self.blobs(self.outputs{n}).set_diff(output_diff{n});
      end
      self.backward_prefilled();
      % retrieve diff from input blobs
      res = cell(length(self.inputs), 1);
      for n = 1:length(self.inputs)
        res{n} = self.blobs(self.inputs{n}).get_diff();
      end
    end
    function copy_from(self, weights_file)
      CHECK(ischar(weights_file), 'weights_file must be a string');
      CHECK_FILE_EXIST(weights_file);
      caffe_('net_copy_from', self.hNet_self, weights_file);
    end
    function reshape(self)
      caffe_('net_reshape', self.hNet_self);
    end
    function save(self, weights_file)
      CHECK(ischar(weights_file), 'weights_file must be a string');
      caffe_('net_save', self.hNet_self, weights_file);
    end
    function set_mean(self,mean)
        self.mean = mean;
    end
  end
end
