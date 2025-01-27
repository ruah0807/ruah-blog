const getDateString = (fileName: string) => {
  const filteredString = fileName.split("/")[1].split("-").slice(-1);
  const dateArray = fileName.split("/")[1].split("-");
  const date = dateArray
    .filter((item) => !filteredString.includes(item))
    .join("-");
  return date;
};

export default getDateString;
