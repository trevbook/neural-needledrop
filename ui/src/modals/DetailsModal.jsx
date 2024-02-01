import React from 'react';
import { Modal, Text, Group, Button } from '@mantine/core';

function DetailsModal({ opened }) {
  return (
    <Modal
      opened={opened}
    >
      <div>
        This is the details modal
      </div>
    </Modal>
  );
}

export default DetailsModal;