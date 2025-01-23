import React from 'react';
import './Header.css';
import logo from '../../public/ruah_blog.svg';
interface HeaderProps {
  toggleSidebar: () => void;
}
const Header: React.FC<HeaderProps> = ({ toggleSidebar }) => {
  const handleTitleClick = () => {
    window.location.href = '/ruah-blog/'; // Navigate to the main page
  };
  return (
    <header>
      <div className='header-title' onClick={handleTitleClick} style={{ cursor: 'pointer' }}>
        <img className='header-logo' draggable="false" src={logo} alt="logo" />
        </div>
      <nav>
        <ul>
          <li>About Me</li>
        <li className='hamburger' onClick={toggleSidebar}>
          ☰
        </li>
        </ul>
      </nav>
    </header>
  );
};

export default Header; 