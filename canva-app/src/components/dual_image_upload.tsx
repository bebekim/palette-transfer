import { useState, useRef } from "react";
import { Button, Columns, Rows, Text, Title } from "@canva/app-ui-kit";
import { FormattedMessage } from "react-intl";
import { transferPalette } from "src/api";
import * as styles from "styles/components.css";

interface UploadedImage {
  file: File;
  url: string;
  name: string;
}

interface DualImageUploadProps {
  onImagesSelected?: (imageA: UploadedImage | null, imageB: UploadedImage | null) => void;
}

export const DualImageUpload = ({ onImagesSelected }: DualImageUploadProps) => {
  const [imageA, setImageA] = useState<UploadedImage | null>(null);
  const [imageB, setImageB] = useState<UploadedImage | null>(null);
  const [transferDirection, setTransferDirection] = useState<'A_to_B' | 'B_to_A'>('A_to_B');
  
  const fileInputARef = useRef<HTMLInputElement>(null);
  const fileInputBRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (file: File, slot: 'A' | 'B') => {
    const url = URL.createObjectURL(file);
    const uploadedImage: UploadedImage = {
      file,
      url,
      name: file.name
    };

    if (slot === 'A') {
      setImageA(uploadedImage);
      onImagesSelected?.(uploadedImage, imageB);
    } else {
      setImageB(uploadedImage);
      onImagesSelected?.(imageA, uploadedImage);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>, slot: 'A' | 'B') => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      handleImageUpload(file, slot);
    }
  };

  const triggerFileSelect = (slot: 'A' | 'B') => {
    if (slot === 'A') {
      fileInputARef.current?.click();
    } else {
      fileInputBRef.current?.click();
    }
  };

  const [isTransferring, setIsTransferring] = useState(false);
  const [transferResult, setTransferResult] = useState<string | null>(null);

  const handleTransfer = async () => {
    if (imageA && imageB) {
      setIsTransferring(true);
      try {
        const sourceImage = transferDirection === 'A_to_B' ? imageA.file : imageB.file;
        const targetImage = transferDirection === 'A_to_B' ? imageB.file : imageA.file;
        
        const result = await transferPalette({
          sourceImage,
          targetImage,
          direction: transferDirection
        });
        
        setTransferResult(result.processedImageUrl);
      } catch (error) {
        console.error('Transfer failed:', error);
      } finally {
        setIsTransferring(false);
      }
    }
  };

  return (
    <Rows spacing="3u">
      <Title size="medium">
        <FormattedMessage
          defaultMessage="Skin Tone Analysis"
          description="Title for the skin tone analysis feature"
        />
      </Title>
      
      <Text>
        <FormattedMessage
          defaultMessage="Upload two photos to analyze and transfer skin tone palettes"
          description="Instructions for the dual image upload"
        />
      </Text>

      <Columns spacing="2u">
        {/* Image Slot A */}
        <Rows spacing="1u">
          <Text size="small" variant="bold">
            <FormattedMessage defaultMessage="Photo A (Before)" />
          </Text>
          <div
            style={{
              width: '200px',
              height: '200px',
              border: '2px dashed #ccc',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              backgroundImage: imageA ? `url(${imageA.url})` : 'none',
              backgroundSize: 'cover',
              backgroundPosition: 'center'
            }}
            onClick={() => triggerFileSelect('A')}
          >
            {!imageA && (
              <Text size="small">
                <FormattedMessage defaultMessage="Click to upload" />
              </Text>
            )}
          </div>
          <input
            ref={fileInputARef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => handleFileChange(e, 'A')}
          />
        </Rows>

        {/* Image Slot B */}
        <Rows spacing="1u">
          <Text size="small" variant="bold">
            <FormattedMessage defaultMessage="Photo B (After)" />
          </Text>
          <div
            style={{
              width: '200px',
              height: '200px',
              border: '2px dashed #ccc',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              backgroundImage: imageB ? `url(${imageB.url})` : 'none',
              backgroundSize: 'cover',
              backgroundPosition: 'center'
            }}
            onClick={() => triggerFileSelect('B')}
          >
            {!imageB && (
              <Text size="small">
                <FormattedMessage defaultMessage="Click to upload" />
              </Text>
            )}
          </div>
          <input
            ref={fileInputBRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => handleFileChange(e, 'B')}
          />
        </Rows>
      </Columns>

      {/* Direction Controls - Centered below images */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '12px', alignItems: 'center' }}>
        <Button
          variant="primary"
          disabled={true}
          style={{ minWidth: '80px' }}
        >
          {transferDirection === 'A_to_B' ? 'A → B' : 'B → A'}
        </Button>
        <Button
          variant="secondary"
          onClick={() => setTransferDirection(transferDirection === 'A_to_B' ? 'B_to_A' : 'A_to_B')}
        >
          <FormattedMessage defaultMessage="Reverse Direction" />
        </Button>
      </div>

      {/* Transfer Button */}
      <Button
        variant="primary"
        onClick={handleTransfer}
        disabled={!imageA || !imageB || isTransferring}
        style={{ alignSelf: 'center' }}
      >
        {isTransferring ? (
          <FormattedMessage defaultMessage="Processing..." />
        ) : (
          <FormattedMessage
            defaultMessage="Transfer Palette"
            description="Button to start the palette transfer process"
          />
        )}
      </Button>

      {/* Transfer Result */}
      {transferResult && (
        <Rows spacing="1u">
          <Text size="medium" variant="bold">
            <FormattedMessage defaultMessage="Transfer Result:" />
          </Text>
          <div
            style={{
              width: '300px',
              height: '300px',
              border: '2px solid #ccc',
              borderRadius: '8px',
              backgroundImage: `url(${transferResult})`,
              backgroundSize: 'cover',
              backgroundPosition: 'center',
              alignSelf: 'center'
            }}
          />
        </Rows>
      )}
    </Rows>
  );
};