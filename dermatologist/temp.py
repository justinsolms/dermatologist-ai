#  Unused stuff goes in this file.

# %% Too many Melanocytic nevi (dx is 'nv')
# Get all nevi
nv = data[data.dx == 'nv']
# Replace all nevi with a random sub-sample of nevi
data = pd.concat([
    data.drop(nv.index), # Drop all nevi from data
    nv.sample(1500, random_state=0),  # Random nevi sample
    ], axis='index', sort=False)


    def stuff(self):

        # #  Reduce the data set size for sane runtimes during unit testing. Use
        # #  stratification to we get a fair representation of categories.
        if unittest_size is not None:
            data, _ = train_test_split(
                data, train_size=unittest_size,
                random_state=self.random_state, shuffle=True,
                stratify=data.classification)

        # Load images
        data['image_path'] = data.image_id.map(
            lambda file: os.path.join(images_dir, file + '.jpg'))

        # Load images
        image_data = list()
        logger.info('Loading and resizing original images.')
        for path in data.image_path:
            image = Image.open(path)
            image_data.append((
                image.resize(image_size, PIL.Image.ANTIALIAS),
                (image.mode, image.width, image.height)
                ))
        image_data = pd.DataFrame.from_records(image_data)
        image_data.columns = ['image', 'image_meta']
        data = pd.concat([data, image_data], axis='columns', sort=False)

        return data

    def save_data(self, test_size=0.2):
        train_path = os.path.join(self.data_dir, 'train.pkl')
        test_path = os.path.join(self.data_dir, 'test.pkl')
        category_path = os.path.join(self.data_dir, 'category.pkl')

        # PIL Images to 4D tensor (n_samples, height, width, channels) with
        # normalized values
        logger.info('Converting PIL images to tensor of uint8.')
        images = np.array(list( map(np.array, self.data_set.image.tolist()) ))
        # logger.info('Deleting PIL images.')
        # del self.data_set['image']
        logger.info('Converting images tensor to float16.')
        images = images.astype(np.float16)
        logger.info('Normalizing image tensor.')
        images /= 255

        # One-hot label encoding of image classification
        logger.info('Creating one-hot classification label tensor.')
        labels_str = self.data_set.classification.values.reshape(-1, 1)
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(labels_str)
        labels = encoder.transform(labels_str)
        categories = encoder.categories_[0]

        # Split data sets, stratify on classes.
        logger.info('Split data into train and test sets.')
        data = images
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=test_size,
            random_state=self.random_state, shuffle=True,
            stratify=labels)

        # Serialize data set
        logger.info('Writing train data set.')
        with open(train_path, 'wb') as stream:
            pickle.dump((train_labels, train_data), stream)
        logger.info('Writing test data set.')
        with open(test_path, 'wb') as stream:
            pickle.dump((test_labels, test_data), stream)
        logger.info('Writing data categories.')
        with open(category_path, 'wb') as stream:
            pickle.dump(categories, stream)

