import { ReactNode } from "react";

const renderHeading = ({
  children,
  headingNo,
}: {
  children: ReactNode;
  headingNo: 1 | 2 | 3;
}) => {
  const text = () => {
    if (typeof children === "string") {
      return children;
    }
    return "";
  };

  if (headingNo === 1) {
    return <h1 id={text().toLowerCase().replace(/\s+/g, "-")}>{children}</h1>;
  } else if (headingNo === 2) {
    return <h2 id={text().toLowerCase().replace(/\s+/g, "-")}>{children}</h2>;
  } else if (headingNo === 3) {
    return <h3 id={text().toLowerCase().replace(/\s+/g, "-")}>{children}</h3>;
  }
};

export default renderHeading;
