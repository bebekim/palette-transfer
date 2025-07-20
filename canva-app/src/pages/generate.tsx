import { Rows } from "@canva/app-ui-kit";
import { AppError, DualImageUpload } from "src/components";

export const GeneratePage = () => (
  <Rows spacing="1u">
    <AppError />
    <DualImageUpload />
  </Rows>
);
